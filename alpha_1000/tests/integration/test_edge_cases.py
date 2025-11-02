"""Edge case tests."""

from __future__ import annotations

from ...engine.rules import load_rules
from ...engine.state import create_initial_state


def test_initial_bombs_available() -> None:
    rules = load_rules()
    state = create_initial_state(rules)
    assert state.bombs_remaining[0] == rules.bombing.bombs_per_player
