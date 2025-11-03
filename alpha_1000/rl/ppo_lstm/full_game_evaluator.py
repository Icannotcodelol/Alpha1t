"""Full game evaluator - tests on complete games to 1000 points."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...bots.base_bot import BotBase
from ...bots.greedy_bot import GreedyBot
from ...bots.heuristic_bot import HeuristicBot
from ...bots.random_bot import RandomBot
from ...engine.game import TysiacGame
from ...engine.state import PlayerID
from ..encoding import encode_action_mask, encode_state
from .agent import PpoLstmAgent

__all__ = ["FullGameEvaluator"]


@dataclass
class FullGameEvaluator:
    """Evaluates agent on complete games to 1000 points."""

    agent: PpoLstmAgent

    def evaluate_full_games(self, games: int = 20, verbose: bool = True) -> dict[str, float]:
        """Evaluate agent playing complete games vs baseline bots.
        
        Args:
            games: Number of complete games per opponent
            verbose: Print progress
            
        Returns:
            Dictionary with win rates and average scores
        """
        
        bots = {
            'random': RandomBot(),
            'greedy': GreedyBot(),
        }
        
        results = {}
        
        for bot_name, bot in bots.items():
            if verbose:
                print(f"Evaluating vs {bot_name}...")
            
            wins = 0
            agent_scores = []
            bot_scores = []
            
            for game_num in range(games):
                if verbose and (game_num + 1) % 5 == 0:
                    print(f"  Game {game_num+1}/{games}...", flush=True)
                
                # Play complete game
                won, agent_score, bot_score = self._play_complete_game(bot, seed=game_num)
                
                if won:
                    wins += 1
                agent_scores.append(agent_score)
                bot_scores.append(bot_score)
            
            win_rate = wins / games
            avg_agent_score = sum(agent_scores) / games
            avg_bot_score = sum(bot_scores) / games
            
            results[f'winrate_vs_{bot_name}'] = win_rate
            results[f'avg_score_vs_{bot_name}'] = avg_agent_score
            results[f'opponent_avg_score_{bot_name}'] = avg_bot_score
            
            if verbose:
                print(f"  Results: {win_rate*100:.1f}% win rate, "
                      f"avg score {avg_agent_score:.0f} vs {avg_bot_score:.0f}")
        
        return results

    def _play_complete_game(self, opponent: BotBase, seed: int = 0) -> tuple[bool, int, int]:
        """Play one complete game to 1000 points.
        
        Returns:
            (agent_won, agent_final_score, opponent_final_score)
        """
        
        game = TysiacGame.new(seed=seed)
        self.agent.network.eval()
        
        hand_count = 0
        max_hands = 50  # Safety limit
        
        while not game.is_finished() and hand_count < max_hands:
            hand_count += 1
            self._play_one_hand(game, opponent)
        
        agent_score = game.state.scores.get(0, 0)
        opponent_score = game.state.scores.get(1, 0)
        agent_won = agent_score >= 1000
        
        return agent_won, agent_score, opponent_score

    def _play_one_hand(self, game: TysiacGame, opponent: BotBase) -> None:
        """Play one hand of the game."""
        
        game.deal()
        state = game.state
        
        # Simple bidding
        game.bidding.start(state)
        leader = (state.dealer + 1) % 2
        start_bid = game.rules.bidding.start_bid
        
        game.bidding.place_bid(state, leader, start_bid)
        game.bidding.place_bid(state, (leader + 1) % 2, None)
        state.playing_player = leader
        state.current_bid = start_bid
        
        # Musik
        game.musik.reveal(state, 0)
        player = state.playing_player or 0
        if len(state.hands[player]) > 10:
            for _ in range(2):
                if state.hands[player]:
                    state.discard_pile.append(state.hands[player].pop(0))
        
        # Play tricks
        turn = state.playing_player or 0
        step_count = 0
        
        while any(state.hands[p] for p in (0, 1)) and step_count < 100:
            step_count += 1
            
            if not state.hands[turn]:
                turn = (turn + 1) % 2
                continue
            
            legal_cards = game.play.legal_cards(state, turn)
            if not legal_cards:
                turn = (turn + 1) % 2
                continue
            
            if turn == 0:  # Agent
                obs = encode_state(state, 0)
                mask, index_map = encode_action_mask(state.hands[0], legal_cards)
                
                with torch.no_grad():
                    output = self.agent.act(obs, {"play": mask}, greedy=True)
                
                card_action = int(output.actions["play"].item())
                card_index = index_map[card_action] if 0 <= card_action < len(index_map) else 0
                
                if card_index < 0 or card_index >= len(state.hands[0]):
                    card_index = 0
            
            else:  # Opponent bot
                action = opponent.select_action(game, 1)
                card_index = action.card_index
                if card_index < 0 or card_index >= len(state.hands[1]):
                    card_index = 0
            
            # Play card
            game.play.play_card(state, turn, card_index)
            
            # Next turn
            if state.trick_history and len(state.trick_history[-1]) == 2:
                turn = state.playing_player or turn
            else:
                turn = (turn + 1) % 2
        
        # Score the hand
        game.scoring.score_hand(state, contract=state.current_bid or game.rules.bidding.start_bid)

