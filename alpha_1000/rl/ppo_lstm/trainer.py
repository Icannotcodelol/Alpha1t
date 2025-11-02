"""Training loop for PPO-LSTM agent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict

import torch

from .agent import PpoLstmAgent
from .buffer import RolloutBuffer
from .selfplay import SelfPlayConfig, SelfPlayWorker

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

    def run_iteration(self, iteration: int) -> None:
        """Collect rollouts, compute advantages, and update the policy."""

        config = self.curriculum.config_for(iteration)
        worker = SelfPlayWorker(config=config, seed=iteration)
        self.buffer.clear()
        worker.run(self.agent, self.buffer)
        last_value = torch.zeros(1, dtype=torch.float32)
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        for _ in range(self.epochs):
            for batch in self.buffer.iter_batches(self.batch_size):
                loss = self._ppo_loss(batch)
                self.agent.update(loss)

    def _ppo_loss(self, batch) -> torch.Tensor:
        """Compute PPO clipped policy and value loss for a batch."""

        observations = self.buffer.stack_observations(batch)
        observations = {key: value.to(self.agent.device) for key, value in observations.items()}
        outputs = self.agent.network(observations)
        advantages = self.buffer.stack_advantages(batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = self.buffer.stack_returns(batch).to(self.agent.device)
        old_log_probs = self.buffer.stack_log_probs(batch, "play").to(self.agent.device)
        actions = self.buffer.stack_actions(batch, "play").to(self.agent.device)
        new_dist = torch.distributions.Categorical(logits=outputs["play_logits"])
        new_log_probs = new_dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -(torch.min(ratio * advantages, clipped)).mean()
        values = outputs["value"]
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy = new_dist.entropy().mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return total_loss


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args(argv)
    agent = PpoLstmAgent.create(learning_rate=args.learning_rate)
    buffer = RolloutBuffer()
    curriculum = CurriculumManager()
    trainer = Trainer(agent=agent, buffer=buffer, curriculum=curriculum)
    trainer.run_iteration(iteration=0)


if __name__ == "__main__":
    main()
