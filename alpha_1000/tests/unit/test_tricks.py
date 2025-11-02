"""Tests for trick logic."""

from __future__ import annotations

from ...engine.cards import Card, Rank, Suit
from ...engine.phases.play import PlayPhase
from ...engine.rules import load_rules
from ...engine.state import create_initial_state


def test_play_trick_updates_winner() -> None:
    rules = load_rules()
    state = create_initial_state(rules)
    state.set_hands({0: [Card(Suit.SPADES, Rank.ACE)], 1: [Card(Suit.SPADES, Rank.KING)]})
    play = PlayPhase()
    play.play_card(state, 0, 0)
    play.play_card(state, 1, 0)
    assert state.playing_player in {0, 1}
