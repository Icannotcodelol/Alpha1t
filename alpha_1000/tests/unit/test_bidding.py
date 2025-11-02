"""Bidding tests."""

from __future__ import annotations

import pytest

from ...engine.cards import Card, Rank, Suit
from ...engine.phases.bidding import BiddingPhase
from ...engine.rules import load_rules
from ...engine.state import GameState, create_initial_state


def make_state() -> GameState:
    rules = load_rules()
    state = create_initial_state(rules)
    state.set_hands({0: [Card(Suit.HEARTS, Rank.KING), Card(Suit.HEARTS, Rank.QUEEN)], 1: []})
    return state


def test_bid_increment_validation() -> None:
    state = make_state()
    bidding = BiddingPhase()
    bidding.start(state)
    bidding.place_bid(state, 0, 100)
    with pytest.raises(Exception):
        bidding.place_bid(state, 1, 105)


def test_proof_validation_success() -> None:
    state = make_state()
    bidding = BiddingPhase()
    bidding.start(state)
    bidding.place_bid(state, 0, 130)
    state.playing_player = 0
    state.current_bid = 130
    assert bidding.challenge(state, 1, 0)
