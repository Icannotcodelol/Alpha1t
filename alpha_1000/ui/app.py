"""Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from ..engine.game import TysiacGame
from ..engine.rules import load_rules
from ..bots.random_bot import RandomBot
from ..bots.greedy_bot import GreedyBot
from ..bots.heuristic_bot import HeuristicBot

BOT_MAP = {
    "Random": RandomBot,
    "Greedy": GreedyBot,
    "Heuristic": HeuristicBot,
}


def main() -> None:
    """Run the Streamlit UI."""

    st.set_page_config(page_title="Alpha-1000", layout="wide")
    st.title("Alpha-1000")
    mode = st.sidebar.selectbox("Mode", ["Human vs Bot", "Bot vs Bot"])
    bot_choice = st.sidebar.selectbox("Bot", list(BOT_MAP.keys()))
    rules = load_rules()
    st.sidebar.json(rules.model_dump())
    st.write("Game state snapshot")
    game = TysiacGame.new(rules=rules)
    game.run_hand()
    st.write(game.state.scores)


if __name__ == "__main__":
    main()
