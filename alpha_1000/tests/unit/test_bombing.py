"""Bombing tests."""

from __future__ import annotations

import pytest

from ...engine.exceptions import BombingError
from ...engine.phases.musik import MusikPhase
from ...engine.rules import load_rules
from ...engine.state import create_initial_state


def test_bombing_limit_enforced() -> None:
    rules = load_rules()
    state = create_initial_state(rules)
    musik = MusikPhase()
    state.bombs_remaining[0] = 1
    musik.bomb(state, 0, hand_index=0, reason="Test")
    with pytest.raises(BombingError):
        musik.bomb(state, 0, hand_index=1, reason="Exhausted")
