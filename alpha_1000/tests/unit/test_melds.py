"""Tests for marriage utilities."""

from __future__ import annotations

from ...engine.cards import Card, Rank, Suit
from ...engine.marriages import MARRIAGE_POINTS, MarriageState, find_marriages


def test_find_marriages_detects_pairs() -> None:
    cards = [Card(Suit.SPADES, Rank.KING), Card(Suit.SPADES, Rank.QUEEN)]
    marriages = find_marriages(cards)
    assert marriages and marriages[0][0] == Suit.SPADES


def test_marriage_state_prevents_duplicates() -> None:
    state = MarriageState(declared={})
    assert state.can_declare(0, Suit.SPADES)
    state.record(0, Suit.SPADES)
    assert not state.can_declare(0, Suit.SPADES)
