"""Action utilities for the Alpha-1000 engine."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List

from .cards import Card
from .exceptions import InvalidActionError

__all__ = ["ActionMask", "mask_playable_cards"]


@dataclass
class ActionMask:
    """Binary mask describing which actions are available."""

    values: List[int]

    def __post_init__(self) -> None:
        if any(val not in {0, 1} for val in self.values):
            msg = "Action masks must contain only 0 or 1 entries"
            raise ValueError(msg)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ActionMask(values={self.values})"

    def as_numpy(self):  # type: ignore[override]
        """Return the mask as a numpy array."""

        import numpy as np

        return np.array(self.values, dtype=np.int8)


@lru_cache(maxsize=1024)
def _card_index(card: Card) -> int:
    """Return a stable index for the given card."""

    from .cards import RANK_ORDER, Suit

    suit_index = list(Suit).index(card.suit)
    rank_index = RANK_ORDER.index(card.rank)
    return suit_index * len(RANK_ORDER) + rank_index


def mask_playable_cards(hand: Iterable[Card], legal: Iterable[Card]) -> ActionMask:
    """Create an action mask for cards that can legally be played."""

    hand_list = list(hand)
    legal_set = {_card_index(card) for card in legal}
    mask = []
    for card in hand_list:
        idx = _card_index(card)
        mask.append(1 if idx in legal_set else 0)
    if not any(mask):
        raise InvalidActionError("No legal cards available")
    return ActionMask(values=mask)
