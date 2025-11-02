"""Hand display utilities."""

from __future__ import annotations

from typing import Iterable

import streamlit as st

from ...engine.cards import Card


def render_hand(cards: Iterable[Card]) -> None:
    """Render a list of cards in the UI."""

    cols = st.columns(len(list(cards)) or 1)
    for col, card in zip(cols, cards):
        col.write(repr(card))
