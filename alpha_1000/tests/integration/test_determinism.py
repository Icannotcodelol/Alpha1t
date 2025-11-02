"""Determinism tests."""

from __future__ import annotations

from ...engine.game import TysiacGame


def test_same_seed_produces_same_scores() -> None:
    game_a = TysiacGame.new(seed=123)
    game_b = TysiacGame.new(seed=123)
    game_a.run_hand()
    game_b.run_hand()
    assert game_a.state.scores == game_b.state.scores
