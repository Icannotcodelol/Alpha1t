"""Card abstractions for the Alpha-1000 Tysiąc engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Tuple

import numpy as np

__all__ = ["Suit", "Rank", "Card", "CARD_POINTS", "create_deck"]


class Suit(str, Enum):
    """Enumeration of the four suits used in Tysiąc."""

    SPADES = "♠"
    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Suit({self.value})"


class Rank(str, Enum):
    """Enumeration of the ranks in order from lowest to highest."""

    NINE = "9"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    TEN = "10"
    ACE = "A"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Rank({self.value})"


RANK_ORDER: List[Rank] = [
    Rank.NINE,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.TEN,
    Rank.ACE,
]

CARD_POINTS = {
    Rank.NINE: 0,
    Rank.JACK: 2,
    Rank.QUEEN: 3,
    Rank.KING: 4,
    Rank.TEN: 10,
    Rank.ACE: 11,
}


@dataclass(frozen=True)
class Card:
    """Immutable representation of a single card."""

    suit: Suit
    rank: Rank

    def __post_init__(self) -> None:
        if self.rank not in CARD_POINTS:
            msg = f"Unknown rank: {self.rank}"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Card({self.rank.value}{self.suit.value})"

    @property
    def points(self) -> int:
        """Return the point value associated with the card."""

        return CARD_POINTS[self.rank]

    def encode(self) -> np.ndarray:
        """Encode the card as a one-hot numpy array."""

        suit_index = list(Suit).index(self.suit)
        rank_index = RANK_ORDER.index(self.rank)
        vec = np.zeros(24, dtype=np.float32)
        vec[suit_index * len(RANK_ORDER) + rank_index] = 1.0
        return vec


def create_deck() -> Tuple[Card, ...]:
    """Create a tuple representing the standard 24-card deck."""

    return tuple(Card(suit=suit, rank=rank) for suit in Suit for rank in RANK_ORDER)


def sort_cards(cards: Iterable[Card]) -> List[Card]:
    """Return cards sorted by suit then rank following game order."""

    suit_order = {suit: idx for idx, suit in enumerate(Suit)}
    rank_order = {rank: idx for idx, rank in enumerate(RANK_ORDER)}
    return sorted(cards, key=lambda card: (suit_order[card.suit], rank_order[card.rank]))


__all__.append("sort_cards")
