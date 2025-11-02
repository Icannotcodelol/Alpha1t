"""Tests for card module."""

from __future__ import annotations

import numpy as np

from ...engine.cards import CARD_POINTS, Card, Rank, Suit, create_deck, sort_cards


def test_card_points_mapping() -> None:
    """Ensure card points mapping is correct."""

    assert CARD_POINTS[Rank.ACE] == 11
    assert CARD_POINTS[Rank.TEN] == 10


def test_create_deck_size() -> None:
    """Deck should contain 24 unique cards."""

    deck = create_deck()
    assert len(deck) == 24
    assert len(set(deck)) == 24


def test_card_encoding_dimension() -> None:
    """Card encoding should produce 24-dim vector."""

    card = Card(Suit.SPADES, Rank.ACE)
    vec = card.encode()
    assert vec.shape == (24,)
    assert np.isclose(vec.sum(), 1.0)


def test_sort_cards_orders_by_suit_then_rank() -> None:
    """Sorting should be deterministic."""

    cards = [Card(Suit.HEARTS, Rank.NINE), Card(Suit.CLUBS, Rank.ACE)]
    sorted_cards = sort_cards(cards)
    assert sorted_cards[0].suit == Suit.CLUBS
