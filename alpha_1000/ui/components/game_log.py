"""Game log component."""

from __future__ import annotations

import streamlit as st

from ...engine.state import GameState


def render_log(state: GameState) -> None:
    """Render a textual log."""

    st.subheader("Game Log")
    st.json({
        "melds": [(player, repr(card_a), repr(card_b)) for player, card_a, card_b in state.meld_history],
        "bombs": [record.__dict__ for record in state.bomb_events],
    })
