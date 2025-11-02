"""Random baseline bot."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .base_bot import BotAction, BotBase


@dataclass
class RandomBot(BotBase):
    """Randomly selects a legal card."""

    name: str = "random"
    rng: random.Random = random.Random()

    def select_action(self, game, player: int) -> BotAction:  # type: ignore[override]
        """Choose a random card index from the player's hand."""

        hand = game.state.hands[player]
        if not hand:
            return BotAction(card_index=0)
        index = self.rng.randrange(len(hand))
        return BotAction(card_index=index)
