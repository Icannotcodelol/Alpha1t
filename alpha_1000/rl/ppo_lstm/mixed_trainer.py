"""Mixed-opponent A2C trainer for better generalization."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ..curriculum import CurriculumSchedule, create_default_curriculum, create_fast_curriculum
from .agent import PpoLstmAgent
from .buffer import RolloutBuffer, Transition
from .evaluator import Evaluator
from .mixed_opponent_selfplay import MixedOpponentConfig, MixedOpponentWorker

__all__ = ["MixedOpponentTrainer", "main"]


@dataclass
class MixedOpponentTrainer:
    """A2C trainer using mixed opponents for better generalization."""

    agent: PpoLstmAgent
    buffer: RolloutBuffer
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    batch_size: int = 32
    log_writer: object | None = None

    def run_iteration(self, iteration: int, config: MixedOpponentConfig) -> Dict[str, float]:
        """Collect rollouts from mixed opponents and update policy."""

        worker = MixedOpponentWorker(config=config, seed=iteration)
        self.buffer.clear()
        
        # Collect data
        self.agent.network.eval()
        with torch.no_grad():
            worker.run(self.agent, self.buffer)
        
        # Compute advantages
        last_value = torch.zeros(1, dtype=torch.float32)
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        # Train
        self.agent.network.train()
        all_metrics = []
        
        heads = ("play",)  # Start with card play only
        
        for head in heads:
            batches = list(self.buffer.iter_head_batches(head, self.batch_size))
            if not batches:
                continue
                
            for batch in batches:
                if len(batch) < 2:
                    continue
                    
                loss, metrics = self._a2c_loss(batch, head=head)
                
                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                    self.agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.network.parameters(), 
                        self.max_grad_norm
                    )
                    self.agent.optimizer.step()
                    all_metrics.append(metrics)
                    self._log_step_metrics(metrics)
        
        if all_metrics:
            avg_metrics = {
                key: sum(m.get(key, 0.0) for m in all_metrics) / len(all_metrics)
                for key in all_metrics[0].keys()
            }
        else:
            avg_metrics = {}
            
        return avg_metrics

    def _a2c_loss(self, batch: List[Transition], *, head: str) -> Tuple[torch.Tensor | None, Dict[str, float]]:
        """Compute A2C loss."""

        try:
            observations = self.buffer.stack_observations(batch)
            observations = {
                key: value.detach().clone().to(self.agent.device) 
                for key, value in observations.items()
            }
            
            outputs = self.agent.network(observations)
            
            # Advantages
            advantages = self.buffer.stack_advantages(batch).to(self.agent.device)
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()
            advantages = advantages.detach()
            
            # Returns
            returns = self.buffer.stack_returns(batch).to(self.agent.device).flatten()
            
            # Actions
            actions = self.buffer.stack_actions(batch, head).to(self.agent.device).flatten()
            
            # Logits with masking
            logits_key = f"{head}_logits"
            logits = outputs[logits_key]
            
            masks = self.buffer.stack_masks(batch, head)
            if masks is not None:
                masks = masks.to(self.agent.device)
                logits = torch.where(masks > 0, logits, torch.full_like(logits, -1e10))
            
            logits = torch.clamp(logits, -10, 10)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return None, {f"{head}/policy_loss": 0.0, f"{head}/value_loss": 0.0, f"{head}/entropy": 0.0}
            
            # A2C loss
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            policy_loss = -(log_probs * advantages).mean()
            
            values = outputs["value"].flatten()
            value_loss = F.mse_loss(values, returns)
            
            entropy = dist.entropy().mean()
            
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return None, {f"{head}/policy_loss": 0.0, f"{head}/value_loss": 0.0, f"{head}/entropy": 0.0}
            
            metrics = {
                f"{head}/policy_loss": float(policy_loss.item()),
                f"{head}/value_loss": float(value_loss.item()),
                f"{head}/entropy": float(entropy.item()),
            }
            
            return total_loss, metrics
            
        except Exception as e:
            print(f"ERROR in _a2c_loss for {head}: {e}")
            return None, {f"{head}/policy_loss": 0.0, f"{head}/value_loss": 0.0, f"{head}/entropy": 0.0}

    def _log_step_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to TensorBoard."""
        if self.log_writer is None:
            return
        try:
            if not hasattr(self, '_step_counter'):
                self._step_counter = 0
            self._step_counter += 1
            for k, v in metrics.items():
                self.log_writer.add_scalar(k, v, self._step_counter)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for mixed-opponent training."""

    parser = argparse.ArgumentParser(description="Train A2C agent with mixed opponents")
    
    # Optimizer
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy bonus")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping")
    
    # Training schedule
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
    parser.add_argument("--games-per-iter", type=int, default=8, help="Games per iteration")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--full-game", action="store_true", help="Play full games to 1000")
    
    # Mixed opponent ratios
    parser.add_argument("--self-play-ratio", type=float, default=0.5, help="Ratio of self-play games")
    parser.add_argument("--random-ratio", type=float, default=0.25, help="Ratio vs RandomBot")
    parser.add_argument("--greedy-ratio", type=float, default=0.25, help="Ratio vs GreedyBot")
    parser.add_argument("--heuristic-ratio", type=float, default=0.0, help="Ratio vs HeuristicBot")
    
    # Evaluation and checkpoints
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluate every N iterations")
    parser.add_argument("--eval-games", type=int, default=50, help="Games per evaluation")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/checkpoints_mixed"), help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=500, help="Save every N iterations")
    parser.add_argument("--logdir", type=Path, default=None, help="TensorBoard log directory")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")

    args = parser.parse_args(argv)

    # Setup
    device = torch.device(args.device)
    agent = PpoLstmAgent.create(learning_rate=args.learning_rate, device=device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.network.load_state_dict(torch.load(args.resume, map_location=device))
    
    buffer = RolloutBuffer()
    
    # TensorBoard
    log_writer = None
    if args.logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            args.logdir.mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=str(args.logdir))
        except ImportError:
            print("Warning: TensorBoard not available")
    
    # Create trainer
    trainer = MixedOpponentTrainer(
        agent=agent,
        buffer=buffer,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        log_writer=log_writer,
    )
    
    # Mixed opponent config
    mo_config = MixedOpponentConfig(
        games_per_iteration=args.games_per_iter,
        self_play_ratio=args.self_play_ratio,
        random_bot_ratio=args.random_ratio,
        greedy_bot_ratio=args.greedy_ratio,
        heuristic_bot_ratio=args.heuristic_ratio,
        full_game=args.full_game,
    )
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(agent=agent) if args.eval_every > 0 else None
    
    print(f"Mixed-Opponent Training Configuration:")
    print(f"  Self-play: {mo_config.self_play_ratio*100:.0f}%")
    print(f"  vs Random: {mo_config.random_bot_ratio*100:.0f}%")
    print(f"  vs Greedy: {mo_config.greedy_bot_ratio*100:.0f}%")
    print(f"  Full game mode: {mo_config.full_game}")
    print()
    
    # Training loop
    for it in range(args.iterations):
        start = time.perf_counter()
        
        # Run iteration
        metrics = trainer.run_iteration(iteration=args.seed + it, config=mo_config)
        duration = time.perf_counter() - start
        
        # Log
        rewards = [float(t.reward.item()) for t in trainer.buffer.transitions]
        avg_reward = sum(rewards) / max(1, len(rewards))
        
        print(
            f"iter={it:05d} steps={len(trainer.buffer.transitions):5d} "
            f"avg_reward={avg_reward:+.4f} time={duration:.2f}s",
            flush=True,
        )
        
        if log_writer:
            log_writer.add_scalar("train/avg_reward", avg_reward, it)
            log_writer.add_scalar("train/num_steps", len(trainer.buffer.transitions), it)
            for k, v in metrics.items():
                log_writer.add_scalar(f"train/{k}", v, it)
        
        # Evaluation with MORE games for reliability
        if evaluator and args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            eval_metrics = evaluator.evaluate(games=args.eval_games)
            print(f"eval ({args.eval_games} games): {eval_metrics}")
            if log_writer:
                for k, v in eval_metrics.items():
                    log_writer.add_scalar(f"eval/{k}", v, it)
        
        # Checkpointing
        if (it + 1) % args.save_every == 0:
            ckpt_path = args.save_dir / f"agent_{it+1:06d}.pt"
            torch.save(agent.network.state_dict(), ckpt_path)
            torch.save(agent.network.state_dict(), args.save_dir / "agent_latest.pt")
            print(f"saved checkpoint: {ckpt_path}")
    
    if log_writer:
        log_writer.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total iterations: {args.iterations}")
    print(f"Final checkpoint: {args.save_dir / 'agent_latest.pt'}")
    print("\nRun comprehensive evaluation:")
    print(f"  python -c \"from alpha_1000.rl.ppo_lstm.evaluator import Evaluator; ...\"")


if __name__ == "__main__":
    main()

