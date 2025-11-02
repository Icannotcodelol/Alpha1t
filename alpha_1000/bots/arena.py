"""Bot tournament utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..engine.game import TysiacGame
from .base_bot import BotBase

__all__ = ["Arena", "MatchResult"]


@dataclass
class MatchResult:
    """Result of a head-to-head match."""

    bot_a: str
    bot_b: str
    score_a: int
    score_b: int

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MatchResult({self.bot_a} {self.score_a}-{self.score_b} {self.bot_b})"


@dataclass
class Arena:
    """Runs matches between registered bots."""

    bots: Sequence[BotBase]

    def run(self, games: int = 10) -> List[MatchResult]:
        """Play multiple matches cycling through bots."""

        results: List[MatchResult] = []
        for i, bot_a in enumerate(self.bots):
            for bot_b in self.bots[i + 1 :]:
                result = self._play_pair(bot_a, bot_b, games)
                results.append(result)
        return results

    def _play_pair(self, bot_a: BotBase, bot_b: BotBase, games: int) -> MatchResult:
        """Play multiple games between two bots."""

        game = TysiacGame.new()
        bot_a.reset()
        bot_b.reset()
        scores = {bot_a.name: 0, bot_b.name: 0}
        players = {0: bot_a, 1: bot_b}
        for _ in range(games):
            game.run_hand()
            scores[bot_a.name] += game.state.scores[0]
            scores[bot_b.name] += game.state.scores[1]
            bot_a.notify_round_end(game)
            bot_b.notify_round_end(game)
        return MatchResult(bot_a=bot_a.name, bot_b=bot_b.name, score_a=scores[bot_a.name], score_b=scores[bot_b.name])
