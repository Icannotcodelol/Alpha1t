"""Bidding panel components."""

from __future__ import annotations

import streamlit as st

from ...engine.state import GameState


def render_bidding(state: GameState) -> None:
    """Render simple bidding summary."""

    st.subheader("Bidding Summary")
    if state.current_bid is None:
        st.write("No active bid")
    else:
        st.write(f"Highest bid: {state.current_bid}")
        st.write(f"Playing player: {state.playing_player}")
