"""Meld management for Alpha-1000."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .cards import Card, Rank, Suit

__all__ = ["MARRIAGE_POINTS", "MarriageState", "find_marriages"]

MARRIAGE_POINTS: Dict[Suit, int] = {
    Suit.SPADES: 40,
    Suit.CLUBS: 60,
    Suit.DIAMONDS: 80,
    Suit.HEARTS: 100,
}


@dataclass
class MarriageState:
    """Tracks marriages declared by each player."""

    declared: Dict[int, List[Suit]]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MarriageState(declared={self.declared})"

    def can_declare(self, player: int, suit: Suit) -> bool:
        """Return whether the player may declare a marriage in the given suit."""

        return suit not in self.declared.setdefault(player, [])

    def record(self, player: int, suit: Suit) -> None:
        """Record a marriage for the player."""

        if suit not in self.declared.setdefault(player, []):
            self.declared[player].append(suit)


def find_marriages(cards: Iterable[Card]) -> List[Tuple[Suit, Card, Card]]:
    """Return a list of available marriages in the provided cards."""

    cards_by_suit: Dict[Suit, Dict[Rank, Card]] = {}
    for card in cards:
        cards_by_suit.setdefault(card.suit, {})[card.rank] = card
    marriages: List[Tuple[Suit, Card, Card]] = []
    for suit in Suit:
        ranks = cards_by_suit.get(suit, {})
        if Rank.KING in ranks and Rank.QUEEN in ranks:
            marriages.append((suit, ranks[Rank.KING], ranks[Rank.QUEEN]))
    return marriages
