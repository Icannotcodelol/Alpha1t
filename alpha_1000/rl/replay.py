"""Replay persistence utilities for analysing self-play games."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .ppo_lstm.buffer import Transition

__all__ = ["ReplayWriter"]


@dataclass
class ReplayWriter:
    """Serialises transitions from self-play to JSON files."""

    directory: Path
    prefix: str = "replay"
    counter: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ReplayWriter(directory={self.directory})"

    def write(self, transitions: Iterable[Transition]) -> Path:
        """Write transitions to disk and return the file path."""

        payload = [self._serialise_transition(transition) for transition in transitions]
        path = self.directory / f"{self.prefix}_{self.counter:05d}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.counter += 1
        return path

    def _serialise_transition(self, transition: Transition) -> dict:
        """Convert transition into JSON friendly structure."""

        obs = {key: value.tolist() for key, value in transition.observation.tensors.items()}
        actions = {key: float(tensor.item()) for key, tensor in transition.actions.items()}
        log_probs = {key: float(tensor.item()) for key, tensor in transition.log_probs.items()}
        return {
            "observation": obs,
            "actions": actions,
            "log_probs": log_probs,
            "reward": float(transition.reward.item()),
            "done": bool(transition.done.item()),
        }
