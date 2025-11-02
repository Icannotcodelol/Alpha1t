"""Integration test covering a simplified hand."""

from __future__ import annotations

from ...engine.game import TysiacGame


def test_run_hand_completes() -> None:
    game = TysiacGame.new()
    game.run_hand()
    assert isinstance(game.state.scores[0], int)
