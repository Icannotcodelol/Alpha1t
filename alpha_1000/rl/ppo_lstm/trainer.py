"""Training loop for PPO-LSTM agent.

Adds a simple CLI to run multiple iterations with optional replay
serialization and checkpoint saving, so training can be kicked off from
the command line or `scripts/train.sh`.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from .agent import PpoLstmAgent
from .buffer import RolloutBuffer
from .selfplay import SelfPlayConfig, SelfPlayWorker
from .buffer import Transition

__all__ = ["CurriculumManager", "Trainer", "main"]


@dataclass
class CurriculumManager:
    """Simple curriculum adjusting self-play parameters over time."""

    stages: Dict[int, SelfPlayConfig] = field(default_factory=dict)

    def config_for(self, iteration: int) -> SelfPlayConfig:
        """Return configuration matching the given iteration."""

        candidates = [stage for stage in self.stages if stage <= iteration]
        stage = max(candidates) if candidates else 0
        return self.stages.get(stage, SelfPlayConfig())


@dataclass
class Trainer:
    """Coordinate self-play data collection and PPO updates."""

    agent: PpoLstmAgent
    buffer: RolloutBuffer
    curriculum: CurriculumManager
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    batch_size: int = 8
    epochs: int = 4
    log_writer: object | None = None

    def run_iteration(self, iteration: int) -> None:
        """Collect rollouts, compute advantages, and update the policy."""

        config = self.curriculum.config_for(iteration)
        worker = SelfPlayWorker(config=config, seed=iteration)
        self.buffer.clear()
        # Collect data with network in eval mode
        self.agent.network.eval()
        with torch.no_grad():
            worker.run(self.agent, self.buffer)
        
        # Normalize rewards for stability
        self.buffer.normalize_rewards()
        
        last_value = torch.zeros(1, dtype=torch.float32)
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        # Train with network in train mode
        self.agent.network.train()
        heads = ("play", "bid", "bomb")
        for _ in range(self.epochs):
            for head in heads:
                for batch in self.buffer.iter_head_batches(head, self.batch_size):
                    if not batch:
                        continue
                    loss, metrics = self._ppo_loss(batch, head=head)
                    self.agent.update(loss)
                    self._log_step_metrics(metrics)

    def _ppo_loss(self, batch: List[Transition], *, head: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss for a specific policy head over a batch."""

        observations = self.buffer.stack_observations(batch)
        observations = {key: value.detach().clone().to(self.agent.device) for key, value in observations.items()}
        outputs = self.agent.network(observations)
        
        # Get advantages with robust normalization
        advantages = self.buffer.stack_advantages(batch).to(self.agent.device)
        if len(advantages) > 1:
            # Normalize only if we have more than 1 sample
            adv_std = advantages.std()
            if adv_std > 1e-6:  # Only normalize if std is meaningful
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                # If std is too small, just center
                advantages = advantages - advantages.mean()
        
        returns = self.buffer.stack_returns(batch).to(self.agent.device)
        if returns.dim() > 1:
            returns = returns.squeeze()
        
        old_log_probs = self.buffer.stack_log_probs(batch, head).to(self.agent.device)
        if old_log_probs.dim() > 1:
            old_log_probs = old_log_probs.squeeze()
            
        actions = self.buffer.stack_actions(batch, head).to(self.agent.device)
        if actions.dim() > 1:
            actions = actions.squeeze()
        
        logits_key = f"{head}_logits"
        logits = outputs[logits_key].clone()  # Clone to avoid in-place modifications
        
        # Apply stored masks if provided
        masks = self.buffer.stack_masks(batch, head)
        if masks is not None:
            masks = masks.to(self.agent.device)
            # Mask out invalid actions with large negative value
            logits = torch.where(masks > 0, logits, torch.full_like(logits, -1e10))
        
        # Check for NaN/Inf before creating distribution
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: NaN/Inf detected in {head}_logits, skipping batch")
            return torch.tensor(0.0, device=self.agent.device), {
                f"{head}/policy_loss": 0.0,
                f"{head}/value_loss": 0.0,
                f"{head}/entropy": 0.0,
            }
        
        new_dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = new_dist.log_prob(actions)
        
        # Clamp ratio to prevent extreme values
        ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, -5, 5))
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -(torch.min(ratio * advantages, clipped)).mean()
        
        # Value loss with clipping
        values = outputs["value"]
        if values.dim() > 1:
            values = values.squeeze()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        
        entropy = new_dist.entropy().mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        metrics = {
            f"{head}/policy_loss": float(policy_loss.item()),
            f"{head}/value_loss": float(value_loss.item()),
            f"{head}/entropy": float(entropy.item()),
        }
        return total_loss, metrics

    def _log_step_metrics(self, metrics: Dict[str, float]) -> None:
        writer = self.log_writer
        if writer is None:
            return
        try:  # pragma: no cover - logging side-effect
            global _TRAIN_STEP
            _TRAIN_STEP = (_TRAIN_STEP + 1) if "_TRAIN_STEP" in globals() else 0
            for k, v in metrics.items():
                writer.add_scalar(k, v, _TRAIN_STEP)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for PPO-LSTM training."""

    parser = argparse.ArgumentParser(description="Train PPO-LSTM agent with self-play")
    # Optimiser + PPO
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per iteration")
    # Self-play + schedule
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--games-per-iter", type=int, default=8, help="Self-play games per iteration")
    parser.add_argument("--full-game", action="store_true", help="Play until target score in each self-play game")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    # Replays + checkpoints
    parser.add_argument("--store-replays", action="store_true", help="Persist self-play replays")
    parser.add_argument("--replay-dir", type=Path, default=Path("runs/replays"), help="Replay output directory")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/checkpoints"), help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N iterations")
    parser.add_argument("--logdir", type=Path, default=Path("runs/tensorboard"), help="TensorBoard log directory")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate versus baselines every N iterations (0=off)")
    # Device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device")

    args = parser.parse_args(argv)

    device = torch.device(args.device)
    agent = PpoLstmAgent.create(learning_rate=args.learning_rate, device=device)
    buffer = RolloutBuffer()

    # Curriculum: single stage from 0 with CLI-configured self-play settings.
    sp_config = SelfPlayConfig(
        games_per_iteration=args.games_per_iter,
        store_replays=args.store_replays,
        replay_dir=args.replay_dir if args.store_replays else None,
        curriculum_stage=0,
        full_game=args.full_game,
    )
    curriculum = CurriculumManager(stages={0: sp_config})
    # Create optional TensorBoard writer
    writer = None
    try:  # pragma: no cover - external dependency
        from torch.utils.tensorboard import SummaryWriter

        args.logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(args.logdir))
    except Exception:
        writer = None

    trainer = Trainer(
        agent=agent,
        buffer=buffer,
        curriculum=curriculum,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_writer=writer,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.store_replays and args.replay_dir is not None:
        args.replay_dir.mkdir(parents=True, exist_ok=True)

    for it in range(args.iterations):
        start = time.perf_counter()
        trainer.run_iteration(iteration=args.seed + it)
        duration = time.perf_counter() - start
        # Lightweight metrics
        rewards = [float(t.reward.item()) for t in trainer.buffer.transitions]
        avg_reward = sum(rewards) / max(1, len(rewards))
        print(
            f"iter={it:05d} steps={len(trainer.buffer.transitions):5d} "
            f"avg_reward={avg_reward:+.4f} time={duration:.2f}s",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar("train/avg_reward", avg_reward, it)

        # Checkpointing
        if (it + 1) % max(1, args.save_every) == 0:
            ckpt_path = args.save_dir / f"agent_{it+1:06d}.pt"
            torch.save(agent.network.state_dict(), ckpt_path)
            torch.save(agent.network.state_dict(), args.save_dir / "agent_latest.pt")
            print(f"saved checkpoint: {ckpt_path}")

        # Optional evaluation
        if args.eval_every and (it + 1) % args.eval_every == 0:
            try:
                from .evaluator import Evaluator

                evaluator = Evaluator(agent=agent)
                metrics = evaluator.evaluate(games=10)
                print(f"eval: {metrics}")
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"eval/{k}", v, it)
            except Exception as exc:  # pragma: no cover - evaluation is optional
                print(f"eval failed: {exc}")


if __name__ == "__main__":
    main()
