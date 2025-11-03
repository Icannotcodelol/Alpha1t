"""Experience buffer for PPO-LSTM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Tuple

import torch

from ..encoding import Observation

__all__ = ["Transition", "RolloutBuffer"]


@dataclass
class Transition:
    """Single step of experience stored in the rollout buffer."""

    observation: Observation
    actions: Dict[str, torch.Tensor]
    log_probs: Dict[str, torch.Tensor]
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    advantage: torch.Tensor | None = None
    return_: torch.Tensor | None = None
    masks: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Transition(reward={float(self.reward)})"


@dataclass
class RolloutBuffer:
    """Buffer accumulating PPO trajectories with GAE computation."""

    transitions: List[Transition] = field(default_factory=list)
    reward_mean: float = 0.0
    reward_std: float = 1.0
    reward_count: int = 0

    def __repr__(self) -> str:  # pragma: no cover - trivial helper
        return f"RolloutBuffer(size={len(self.transitions)})"
    
    def normalize_rewards(self, beta: float = 0.99) -> None:
        """Normalize rewards using running statistics."""
        if not self.transitions:
            return
        
        # Collect all rewards
        rewards = [float(t.reward.item()) for t in self.transitions]
        
        # Update running mean and std
        for r in rewards:
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            self.reward_std = max(0.01, ((self.reward_std ** 2) * (self.reward_count - 1) + delta * delta2) / self.reward_count) ** 0.5
        
        # Normalize transition rewards
        for t in self.transitions:
            normalized = (float(t.reward.item()) - self.reward_mean) / (self.reward_std + 1e-8)
            t.reward = torch.tensor(normalized, dtype=torch.float32)

    def add(self, transition: Transition) -> None:
        """Append a transition to the buffer."""

        self.transitions.append(transition)

    def clear(self) -> None:
        """Remove all stored transitions."""

        self.transitions.clear()

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ) -> None:
        """Compute GAE advantages for the stored transitions."""

        advantages: List[torch.Tensor] = []
        gae = torch.zeros_like(last_value)
        for step in reversed(self.transitions):
            mask = 1.0 - step.done
            delta = step.reward + gamma * last_value * mask - step.value
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            last_value = step.value
        for transition, advantage in zip(self.transitions, advantages, strict=True):
            transition.advantage = advantage
            transition.return_ = advantage + transition.value

    def iter_batches(self, batch_size: int) -> Iterator[List[Transition]]:
        """Yield mini-batches of transitions."""

        for start in range(0, len(self.transitions), batch_size):
            yield self.transitions[start : start + batch_size]

    def iter_head_batches(self, head: str, batch_size: int) -> Iterator[List[Transition]]:
        """Yield mini-batches filtered to transitions that include `head`."""

        filtered = [t for t in self.transitions if head in t.actions]
        for start in range(0, len(filtered), batch_size):
            yield filtered[start : start + batch_size]

    def stack_observations(self, batch: Iterable[Transition]) -> Dict[str, torch.Tensor]:
        """Stack observation tensors for the given batch."""

        stacked: Dict[str, List[torch.Tensor]] = {}
        for transition in batch:
            for key, array in transition.observation.tensors.items():
                tensor = torch.as_tensor(array, dtype=torch.float32)
                stacked.setdefault(key, []).append(tensor)
        return {key: torch.stack(values, dim=0) for key, values in stacked.items()}

    def stack_actions(self, batch: Iterable[Transition], head: str) -> torch.Tensor:
        """Stack actions for the specified policy head."""

        return torch.stack([transition.actions[head] for transition in batch], dim=0)

    def stack_log_probs(self, batch: Iterable[Transition], head: str) -> torch.Tensor:
        """Stack log probabilities for the specified head."""

        return torch.stack([transition.log_probs[head] for transition in batch], dim=0)

    def stack_masks(self, batch: Iterable[Transition], head: str) -> torch.Tensor | None:
        """Stack action masks for the specified head if present."""

        masks = [transition.masks.get(head) for transition in batch if transition.masks is not None]  # type: ignore[union-attr]
        if not masks or any(m is None for m in masks):
            return None
        return torch.stack([m for m in masks if m is not None], dim=0)  # type: ignore[arg-type]

    def stack_values(self, batch: Iterable[Transition]) -> torch.Tensor:
        """Stack value predictions."""

        return torch.stack([transition.value for transition in batch], dim=0)

    def stack_returns(self, batch: Iterable[Transition]) -> torch.Tensor:
        """Stack Monte Carlo returns."""

        return torch.stack([transition.return_ for transition in batch if transition.return_ is not None], dim=0)

    def stack_advantages(self, batch: Iterable[Transition]) -> torch.Tensor:
        """Stack advantage estimates."""

        return torch.stack([transition.advantage for transition in batch if transition.advantage is not None], dim=0)
