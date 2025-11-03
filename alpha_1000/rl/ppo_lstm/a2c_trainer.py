"""Advantage Actor-Critic (A2C) trainer - more stable than PPO for this task."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ..curriculum import CurriculumSchedule, create_default_curriculum, create_fast_curriculum
from .agent import PpoLstmAgent
from .buffer import RolloutBuffer, Transition
from .evaluator import Evaluator
from .selfplay import SelfPlayConfig
from .stable_selfplay import StableSelfPlayWorker

__all__ = ["A2CTrainer", "main"]


@dataclass
class A2CTrainer:
    """Simpler A2C algorithm - more stable than PPO for card games."""

    agent: PpoLstmAgent
    buffer: RolloutBuffer
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 32
    log_writer: object | None = None
    
    # Reward normalization - track running stats
    reward_mean: float = 0.0
    reward_std: float = 1.0
    reward_count: int = 0

    def run_iteration(self, iteration: int, config: SelfPlayConfig) -> Dict[str, float]:
        """Collect rollouts and update the policy with A2C."""

        worker = StableSelfPlayWorker(config=config, seed=iteration)
        self.buffer.clear()
        
        # Collect data with network in eval mode
        self.agent.network.eval()
        with torch.no_grad():
            worker.run(self.agent, self.buffer)
        
        # Update reward normalization statistics
        self._update_reward_stats()
        
        # Compute returns and advantages
        last_value = torch.zeros(1, dtype=torch.float32)
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        # Train with network in train mode
        self.agent.network.train()
        
        # A2C update - process all data at once (no multi-epoch)
        all_metrics = []
        heads = ("play", "bid", "bomb")
        
        for head in heads:
            batches = list(self.buffer.iter_head_batches(head, self.batch_size))
            if not batches:
                continue
                
            for batch in batches:
                if len(batch) < 2:  # Skip tiny batches
                    continue
                    
                loss, metrics = self._a2c_loss(batch, head=head)
                
                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                    # Update with gradient clipping
                    self.agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.network.parameters(), 
                        self.max_grad_norm
                    )
                    self.agent.optimizer.step()
                    all_metrics.append(metrics)
                    self._log_step_metrics(metrics)
        
        # Aggregate metrics
        if all_metrics:
            avg_metrics = {
                key: sum(m.get(key, 0.0) for m in all_metrics) / len(all_metrics)
                for key in all_metrics[0].keys()
            }
        else:
            avg_metrics = {}
            
        return avg_metrics

    def _update_reward_stats(self) -> None:
        """Update running mean/std of rewards for normalization."""
        rewards = [float(t.reward.item()) for t in self.buffer.transitions]
        if not rewards:
            return
            
        # Exponential moving average
        alpha = 0.01
        batch_mean = sum(rewards) / len(rewards)
        batch_std = (sum((r - batch_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
        self.reward_std = (1 - alpha) * self.reward_std + alpha * max(batch_std, 0.01)
        self.reward_count += len(rewards)

    def _a2c_loss(self, batch: List[Transition], *, head: str) -> Tuple[torch.Tensor | None, Dict[str, float]]:
        """Compute A2C loss - simpler than PPO, more stable."""

        try:
            observations = self.buffer.stack_observations(batch)
            observations = {
                key: value.detach().clone().to(self.agent.device) 
                for key, value in observations.items()
            }
            
            outputs = self.agent.network(observations)
            
            # Get advantages with robust normalization
            advantages = self.buffer.stack_advantages(batch).to(self.agent.device)
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()
            
            # Detach advantages to prevent backprop through them
            advantages = advantages.detach()
            
            # Get returns
            returns = self.buffer.stack_returns(batch).to(self.agent.device).flatten()
            
            # Get old actions
            actions = self.buffer.stack_actions(batch, head).to(self.agent.device).flatten()
            
            # Get logits and apply mask
            logits_key = f"{head}_logits"
            logits = outputs[logits_key]
            
            # Apply action mask
            masks = self.buffer.stack_masks(batch, head)
            if masks is not None:
                masks = masks.to(self.agent.device)
                logits = torch.where(masks > 0, logits, torch.full_like(logits, -1e10))
            
            # Clip logits to prevent overflow
            logits = torch.clamp(logits, -10, 10)
            
            # Check for NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"WARNING: NaN/Inf in {head}_logits, skipping batch")
                return None, {
                    f"{head}/policy_loss": 0.0,
                    f"{head}/value_loss": 0.0,
                    f"{head}/entropy": 0.0,
                }
            
            # Create distribution and compute log probs
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            
            # A2C policy loss (no ratio clipping like PPO)
            policy_loss = -(log_probs * advantages).mean()
            
            # Value loss with clipping
            values = outputs["value"].flatten()
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Check final loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"WARNING: NaN/Inf in total loss for {head}, skipping")
                return None, {
                    f"{head}/policy_loss": 0.0,
                    f"{head}/value_loss": 0.0,
                    f"{head}/entropy": 0.0,
                }
            
            metrics = {
                f"{head}/policy_loss": float(policy_loss.item()),
                f"{head}/value_loss": float(value_loss.item()),
                f"{head}/entropy": float(entropy.item()),
            }
            
            return total_loss, metrics
            
        except Exception as e:
            print(f"ERROR in _a2c_loss for {head}: {e}")
            return None, {
                f"{head}/policy_loss": 0.0,
                f"{head}/value_loss": 0.0,
                f"{head}/entropy": 0.0,
            }

    def _log_step_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to TensorBoard if writer is available."""
        if self.log_writer is None:
            return
        try:
            if not hasattr(self, '_step_counter'):
                self._step_counter = 0
            self._step_counter += 1
            for k, v in metrics.items():
                self.log_writer.add_scalar(k, v, self._step_counter)
        except Exception:
            pass  # Fail silently if logging doesn't work


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for A2C training."""

    parser = argparse.ArgumentParser(description="Train A2C agent with self-play")
    # Optimizer
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate (lower for stability)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy bonus coefficient")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping")
    
    # Self-play + schedule
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--games-per-iter", type=int, default=4, help="Self-play games per iteration (overridden by curriculum)")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--curriculum", type=str, default="default", choices=["default", "fast", "none"], help="Curriculum type")
    
    # Evaluation and checkpoints
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluate every N iterations (0=off)")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/checkpoints"), help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N iterations")
    parser.add_argument("--logdir", type=Path, default=None, help="TensorBoard log directory")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device")

    args = parser.parse_args(argv)

    # Setup
    device = torch.device(args.device)
    agent = PpoLstmAgent.create(learning_rate=args.learning_rate, device=device)
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
    trainer = A2CTrainer(
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
    
    # Setup curriculum
    if args.curriculum == "default":
        curriculum = create_default_curriculum()
        print("Using default curriculum (4 stages)")
    elif args.curriculum == "fast":
        curriculum = create_fast_curriculum()
        print("Using fast curriculum (3 stages)")
    else:
        # No curriculum - use fixed config
        curriculum = None
        print("No curriculum - fixed configuration")
    
    # Setup directories
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluator
    evaluator = Evaluator(agent=agent) if args.eval_every > 0 else None
    
    # Training loop
    for it in range(args.iterations):
        start = time.perf_counter()
        
        # Get config from curriculum or use fixed
        if curriculum:
            sp_config = curriculum.get_config(it)
            # Log stage transitions
            if it > 0:
                prev_stage = curriculum.get_stage(it - 1)
                curr_stage = curriculum.get_stage(it)
                if prev_stage.name != curr_stage.name:
                    print(f"\n>>> Curriculum Stage: {curr_stage.name} - {curr_stage.description}\n")
        else:
            sp_config = SelfPlayConfig(
                games_per_iteration=args.games_per_iter,
                store_replays=False,
            )
        
        # Run iteration
        metrics = trainer.run_iteration(iteration=args.seed + it, config=sp_config)
        
        duration = time.perf_counter() - start
        
        # Log iteration stats
        rewards = [float(t.reward.item()) for t in trainer.buffer.transitions]
        avg_reward = sum(rewards) / max(1, len(rewards))
        
        print(
            f"iter={it:05d} steps={len(trainer.buffer.transitions):5d} "
            f"avg_reward={avg_reward:+.4f} time={duration:.2f}s",
            flush=True,
        )
        
        # Log to TensorBoard
        if log_writer:
            log_writer.add_scalar("train/avg_reward", avg_reward, it)
            log_writer.add_scalar("train/num_steps", len(trainer.buffer.transitions), it)
            for k, v in metrics.items():
                log_writer.add_scalar(f"train/{k}", v, it)
        
        # Evaluation
        if evaluator and args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            eval_metrics = evaluator.evaluate(games=10)
            print(f"eval: {eval_metrics}")
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


if __name__ == "__main__":
    main()

