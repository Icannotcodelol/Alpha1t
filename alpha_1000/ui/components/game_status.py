"""Game status panel."""

from __future__ import annotations

import streamlit as st

from ...engine.state import GameState


def render_status(state: GameState) -> None:
    """Render scores and meta information."""

    st.subheader("Scores")
    for player, score in state.scores.items():
        st.write(f"Player {player}: {score}")
    st.write(f"Dealer: {state.dealer}")
