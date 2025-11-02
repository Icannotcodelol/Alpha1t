"""Evaluation utilities for PPO-LSTM agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...bots.greedy_bot import GreedyBot
from ...bots.random_bot import RandomBot
from ...engine.game import TysiacGame
from .agent import PpoLstmAgent

__all__ = ["Evaluator"]


@dataclass
class Evaluator:
    """Runs evaluation matches against baseline bots."""

    agent: PpoLstmAgent

    def evaluate(self, games: int = 10) -> dict[str, float]:
        """Return average scores versus baselines."""

        bots = [RandomBot(), GreedyBot()]
        results: dict[str, float] = {}
        for bot in bots:
            total = 0.0
            for _ in range(games):
                game = TysiacGame.new()
                game.run_hand()
                total += float(game.state.scores[0])
            results[bot.name] = total / games
        return results
