"""Tests for RL state encoding utilities."""

from __future__ import annotations

import numpy as np

from ...engine.game import TysiacGame
from ...rl.encoding import MAX_HAND_CARDS, CARDS_DIM, encode_action_mask, encode_state


def test_encode_state_produces_expected_shapes() -> None:
    """State encoder should produce tensors with consistent shapes."""

    game = TysiacGame.new()
    game.deal()
    observation = encode_state(game.state, player=0)
    tensors = observation.tensors
    assert tensors["hand"].shape == (MAX_HAND_CARDS, CARDS_DIM)
    assert tensors["hand_mask"].shape == (MAX_HAND_CARDS,)
    assert tensors["current_trick"].shape[1] == CARDS_DIM
    assert tensors["trick_history"].shape[-1] == CARDS_DIM
    assert tensors["scores"].shape == (2,)


def test_encode_action_mask_padding() -> None:
    """Action mask should pad up to the maximum hand size."""

    game = TysiacGame.new()
    game.deal()
    legal_cards = game.play.legal_cards(game.state, player=0)
    mask, index_map = encode_action_mask(game.state.hands[0], legal_cards)
    assert mask.shape == (MAX_HAND_CARDS,)
    assert set(index_map[len(legal_cards):]) == {-1}
    assert np.isclose(mask.sum(), float(len(legal_cards)))
