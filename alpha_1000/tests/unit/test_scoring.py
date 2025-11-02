"""Scoring tests."""

from __future__ import annotations

from ...engine.cards import Card, Rank, Suit
from ...engine.phases.scoring import ScoringPhase
from ...engine.rules import load_rules
from ...engine.state import create_initial_state


def test_scoring_applies_contract() -> None:
    rules = load_rules()
    state = create_initial_state(rules)
    state.playing_player = 0
    state.current_bid = 100
    state.trick_history = [[(0, Card(Suit.SPADES, Rank.ACE)), (1, Card(Suit.HEARTS, Rank.NINE))]]
    scoring = ScoringPhase()
    totals = scoring.score_hand(state, contract=100)
    assert state.scores[0] >= 100
