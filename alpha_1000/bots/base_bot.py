"""Bot interface for Alpha-1000."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..engine.game import TysiacGame

__all__ = ["BotBase", "BotAction"]


@dataclass
class BotAction:
    """Represents a bot decision in the play phase."""

    card_index: int

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"BotAction(card_index={self.card_index})"


class BotBase(ABC):
    """Abstract base class for bot implementations."""

    name: str = "bot"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self.name

    @abstractmethod
    def select_action(self, game: TysiacGame, player: int) -> BotAction:
        """Return the next action to execute."""

    def notify_round_end(self, game: TysiacGame) -> None:
        """Hook called after each hand."""

    def reset(self) -> None:
        """Reset internal state if any."""
