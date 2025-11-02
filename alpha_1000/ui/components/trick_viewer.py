"""Trick viewer component."""

from __future__ import annotations

import streamlit as st

from ...engine.state import GameState


def render_tricks(state: GameState) -> None:
    """Render completed tricks."""

    st.subheader("Tricks")
    if not state.trick_history:
        st.write("No tricks played yet")
        return
    for idx, trick in enumerate(state.trick_history, start=1):
        cards = ", ".join(f"P{player}:{card}" for player, card in trick)
        st.write(f"Trick {idx}: {cards}")
