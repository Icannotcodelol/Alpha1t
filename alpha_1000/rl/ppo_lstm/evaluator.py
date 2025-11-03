"""Evaluation utilities for PPO-LSTM agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from ...bots.greedy_bot import GreedyBot
from ...bots.random_bot import RandomBot
from ...bots.base_bot import BotBase
from ...engine.game import TysiacGame
from ...rl.encoding import encode_action_mask, encode_state
from .agent import PpoLstmAgent

__all__ = ["Evaluator"]


@dataclass
class Evaluator:
    """Runs evaluation matches against baseline bots."""

    agent: PpoLstmAgent

    def evaluate(self, games: int = 10) -> Dict[str, float]:
        """Return win rate vs baseline bots using play-phase matchups.

        Uses minimal bidding/musik setup and focuses on trick play to assess
        whether the agent's play policy improves. Metrics are per-bot win rate.
        """

        bots: Sequence[BotBase] = [RandomBot(), GreedyBot()]
        results: Dict[str, float] = {}
        for bot in bots:
            wins = 0
            for _ in range(games):
                if self._play_hand_vs_bot(bot):
                    wins += 1
            results[f"winrate_vs_{bot.name}"] = wins / games
        return results

    def _play_hand_vs_bot(self, bot: BotBase) -> bool:
        """Play one hand: agent (player 0) vs baseline bot (player 1)."""

        import torch
        self.agent.network.eval()  # Put network in eval mode
        game = TysiacGame.new()
        game.deal()
        state = game.state
        # Minimal bidding: player 0 declares start bid
        game.bidding.start(state)
        leader = (state.dealer + 1) % 2
        start_bid = game.rules.bidding.start_bid
        game.bidding.place_bid(state, leader, start_bid)
        other = (leader + 1) % 2
        game.bidding.place_bid(state, other, None)
        state.playing_player = leader
        state.current_bid = start_bid
        # Play tricks with agent and baseline
        turn = state.playing_player or 0
        while any(state.hands[p] for p in (0, 1)):
            if not state.hands[turn]:
                turn = (turn + 1) % 2
                continue
            if turn == 0:
                legal = game.play.legal_cards(state, 0)
                obs = encode_state(state, 0)
                mask, index_map = encode_action_mask(state.hands[0], legal)
                with torch.no_grad():
                    out = self.agent.act(obs, {"play": mask}, greedy=True)
                idx = int(out.actions["play"].item())
                hand_idx = index_map[idx] if 0 <= idx < len(index_map) and index_map[idx] != -1 else 0
            else:
                act = bot.select_action(game, 1)
                hand_idx = act.card_index
                if hand_idx < 0 or hand_idx >= len(state.hands[1]):
                    hand_idx = 0
            game.play.play_card(state, turn, hand_idx)
            if state.trick_history and len(state.trick_history[-1]) == 2:
                turn = state.playing_player or turn
            else:
                turn = (turn + 1) % 2

        # Score and return if agent scored higher this hand
        game.scoring.score_hand(state, contract=start_bid)
        return state.scores[0] > state.scores[1]
