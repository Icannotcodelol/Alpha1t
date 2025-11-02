"""Heuristic bot implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .base_bot import BotAction, BotBase
from ..engine.cards import CARD_POINTS, Rank


@dataclass
class HeuristicBot(BotBase):
    """Favors trump-like suits and preserves tens."""

    name: str = "heuristic"

    def select_action(self, game, player: int) -> BotAction:  # type: ignore[override]
        """Select an action using a simple heuristic."""

        hand = game.state.hands[player]
        if not hand:
            return BotAction(card_index=0)
        # Prefer discarding low cards, keep tens and aces
        def score(idx: int) -> int:
            card = hand[idx]
            base = CARD_POINTS[card.rank]
            if card.rank == Rank.TEN:
                base -= 5
            if card.rank == Rank.ACE:
                base -= 3
            return base

        best_index = min(range(len(hand)), key=score)
        return BotAction(card_index=best_index)
