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

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Transition(reward={float(self.reward)})"


@dataclass
class RolloutBuffer:
    """Buffer accumulating PPO trajectories with GAE computation."""

    transitions: List[Transition] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - trivial helper
        return f"RolloutBuffer(size={len(self.transitions)})"

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

    def stack_observations(self, batch: Iterable[Transition]) -> Dict[str, torch.Tensor]:
        """Stack observation tensors for the given batch."""

        stacked: Dict[str, List[torch.Tensor]] = {}
        for transition in batch:
            for key, array in transition.observation.tensors.items():
                tensor = torch.as_tensor(array, dtype=torch.float32)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                stacked.setdefault(key, []).append(tensor)
        return {key: torch.stack(values, dim=0) for key, values in stacked.items()}

    def stack_actions(self, batch: Iterable[Transition], head: str) -> torch.Tensor:
        """Stack actions for the specified policy head."""

        return torch.stack([transition.actions[head] for transition in batch], dim=0)

    def stack_log_probs(self, batch: Iterable[Transition], head: str) -> torch.Tensor:
        """Stack log probabilities for the specified head."""

        return torch.stack([transition.log_probs[head] for transition in batch], dim=0)

    def stack_values(self, batch: Iterable[Transition]) -> torch.Tensor:
        """Stack value predictions."""

        return torch.stack([transition.value for transition in batch], dim=0)

    def stack_returns(self, batch: Iterable[Transition]) -> torch.Tensor:
        """Stack Monte Carlo returns."""

        return torch.stack([transition.return_ for transition in batch if transition.return_ is not None], dim=0)

    def stack_advantages(self, batch: Iterable[Transition]) -> torch.Tensor:
        """Stack advantage estimates."""

        return torch.stack([transition.advantage for transition in batch if transition.advantage is not None], dim=0)
