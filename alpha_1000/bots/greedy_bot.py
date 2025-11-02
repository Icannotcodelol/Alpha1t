"""Greedy baseline bot."""

from __future__ import annotations

from dataclasses import dataclass

from .base_bot import BotAction, BotBase
from ..engine.cards import CARD_POINTS


@dataclass
class GreedyBot(BotBase):
    """Plays the highest point card available."""

    name: str = "greedy"

    def select_action(self, game, player: int) -> BotAction:  # type: ignore[override]
        """Play the highest point value card."""

        hand = game.state.hands[player]
        if not hand:
            return BotAction(card_index=0)
        best_index = max(range(len(hand)), key=lambda idx: CARD_POINTS[hand[idx].rank])
        return BotAction(card_index=best_index)
