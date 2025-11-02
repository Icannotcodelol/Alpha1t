"""Fixture helpers for tests."""

from __future__ import annotations

from ...engine.cards import Card, Rank, Suit

PRESET_HAND = [Card(Suit.SPADES, Rank.ACE), Card(Suit.SPADES, Rank.TEN)]
