"""Mixed-opponent training - play against diverse bots for better generalization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import torch

from ...bots.base_bot import BotBase
from ...bots.random_bot import RandomBot
from ...bots.greedy_bot import GreedyBot
from ...bots.heuristic_bot import HeuristicBot
from ...engine.game import TysiacGame
from ...engine.state import GameState, PlayerID
from ..encoding import encode_action_mask, encode_state
from .agent import PpoLstmAgent
from .buffer import RolloutBuffer, Transition
from .selfplay import SelfPlayConfig

__all__ = ["MixedOpponentConfig", "MixedOpponentWorker"]


@dataclass
class MixedOpponentConfig:
    """Configuration for mixed-opponent training."""
    
    games_per_iteration: int = 8
    self_play_ratio: float = 0.5      # 50% self-play
    random_bot_ratio: float = 0.25    # 25% vs random
    greedy_bot_ratio: float = 0.25    # 25% vs greedy
    heuristic_bot_ratio: float = 0.0  # 0% vs heuristic (too hard initially)
    full_game: bool = False
    
    def get_opponent_for_game(self, game_index: int) -> str:
        """Determine opponent type for this game based on ratios."""
        ratios = {
            'self': self.self_play_ratio,
            'random': self.random_bot_ratio,
            'greedy': self.greedy_bot_ratio,
            'heuristic': self.heuristic_bot_ratio,
        }
        
        # Normalize ratios
        total = sum(ratios.values())
        normalized = {k: v/total for k, v in ratios.items()}
        
        # Use game index to deterministically select opponent
        threshold = (game_index % 100) / 100.0
        cumulative = 0.0
        
        for opponent_type, ratio in normalized.items():
            cumulative += ratio
            if threshold < cumulative:
                return opponent_type
        
        return 'self'  # Fallback


@dataclass
class MixedOpponentWorker:
    """Runs mixed-opponent games for better generalization."""

    config: MixedOpponentConfig
    seed: int = 0

    def run(self, agent: PpoLstmAgent, buffer: RolloutBuffer) -> None:
        """Execute games against diverse opponents."""
        
        rng = random.Random(self.seed)
        
        for game_idx in range(self.config.games_per_iteration):
            game = TysiacGame.new(seed=rng.randint(0, 100_000))
            opponent_type = self.config.get_opponent_for_game(game_idx)
            
            if opponent_type == 'self':
                self._play_selfplay_game(agent, game, buffer)
            else:
                opponent_bot = self._create_bot(opponent_type)
                self._play_vs_bot_game(agent, opponent_bot, game, buffer)

    def _create_bot(self, bot_type: str) -> BotBase:
        """Create bot instance by type."""
        if bot_type == 'random':
            return RandomBot()
        elif bot_type == 'greedy':
            return GreedyBot()
        elif bot_type == 'heuristic':
            return HeuristicBot()
        else:
            return RandomBot()

    def _play_selfplay_game(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Play game with agent controlling both players (classic self-play)."""
        
        if self.config.full_game:
            while not game.is_finished():
                self._play_selfplay_hand(agent, game, buffer)
        else:
            self._play_selfplay_hand(agent, game, buffer)

    def _play_vs_bot_game(
        self, agent: PpoLstmAgent, bot: BotBase, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Play game with agent as player 0, bot as player 1."""
        
        if self.config.full_game:
            while not game.is_finished():
                self._play_vs_bot_hand(agent, bot, game, buffer)
        else:
            self._play_vs_bot_hand(agent, bot, game, buffer)

    def _play_selfplay_hand(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Play one hand with self-play."""
        
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
            # Simple discard
            for _ in range(2):
                if state.hands[player]:
                    state.discard_pile.append(state.hands[player].pop(0))
        
        # Play tricks - both players controlled by agent
        self._play_tricks_selfplay(agent, game, buffer)
        
        # Score
        game.scoring.score_hand(state, contract=state.current_bid or start_bid)

    def _play_vs_bot_hand(
        self, agent: PpoLstmAgent, bot: BotBase, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Play one hand: agent (player 0) vs bot (player 1)."""
        
        game.deal()
        state = game.state
        
        # Simple bidding - agent always declares
        game.bidding.start(state)
        game.bidding.place_bid(state, 0, game.rules.bidding.start_bid)
        game.bidding.place_bid(state, 1, None)
        state.playing_player = 0
        state.current_bid = game.rules.bidding.start_bid
        
        # Musik
        game.musik.reveal(state, 0)
        if len(state.hands[0]) > 10:
            for _ in range(2):
                if state.hands[0]:
                    state.discard_pile.append(state.hands[0].pop(0))
        
        # Play tricks - agent vs bot
        self._play_tricks_vs_bot(agent, bot, game, buffer)
        
        # Score
        game.scoring.score_hand(state, contract=state.current_bid or game.rules.bidding.start_bid)

    def _play_tricks_selfplay(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Both players controlled by agent."""
        
        state = game.state
        turn = state.playing_player or 0
        step_count = 0
        
        while any(state.hands[p] for p in (0, 1)) and step_count < 100:
            step_count += 1
            
            if not state.hands[turn]:
                turn = (turn + 1) % 2
                continue
            
            # Get legal actions
            legal_cards = game.play.legal_cards(state, turn)
            if not legal_cards:
                turn = (turn + 1) % 2
                continue
            
            # Agent decision
            observation = encode_state(state, turn)
            mask, index_map = encode_action_mask(state.hands[turn], legal_cards)
            
            agent_output = agent.act(observation, {"play": mask})
            card_action = int(agent_output.actions["play"].item())
            card_index = index_map[card_action] if 0 <= card_action < len(index_map) else 0
            
            if card_index < 0 or card_index >= len(state.hands[turn]):
                card_index = 0
            
            # Save state before action
            prev_trick_count = len(state.trick_history)
            prev_score = state.scores.get(turn, 0)
            
            # Play card
            game.play.play_card(state, turn, card_index)
            
            # Compute reward
            reward_value = self._compute_reward(state, turn, prev_trick_count, prev_score)
            done = not any(state.hands[p] for p in (0, 1))
            
            # Store transition
            transition = Transition(
                observation=observation,
                actions={"play": agent_output.actions["play"]},
                log_probs={"play": agent_output.log_probs["play"]},
                value=agent_output.value.detach(),
                reward=torch.tensor(reward_value, dtype=torch.float32),
                done=torch.tensor(1.0 if done else 0.0, dtype=torch.float32),
                masks={"play": torch.as_tensor(mask, dtype=torch.float32)},
            )
            buffer.add(transition)
            
            # Next turn
            if state.trick_history and len(state.trick_history[-1]) == 2:
                turn = state.playing_player or turn
            else:
                turn = (turn + 1) % 2

    def _play_tricks_vs_bot(
        self, agent: PpoLstmAgent, bot: BotBase, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Agent (player 0) vs Bot (player 1)."""
        
        state = game.state
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
            
            if turn == 0:  # Agent's turn
                # Get observation and action
                observation = encode_state(state, 0)
                mask, index_map = encode_action_mask(state.hands[0], legal_cards)
                
                agent_output = agent.act(observation, {"play": mask})
                card_action = int(agent_output.actions["play"].item())
                card_index = index_map[card_action] if 0 <= card_action < len(index_map) else 0
                
                if card_index < 0 or card_index >= len(state.hands[0]):
                    card_index = 0
                
                # Save state
                prev_trick_count = len(state.trick_history)
                prev_score = state.scores.get(0, 0)
                
                # Play card
                game.play.play_card(state, 0, card_index)
                
                # Compute reward
                reward_value = self._compute_reward(state, 0, prev_trick_count, prev_score)
                done = not any(state.hands[p] for p in (0, 1))
                
                # Store transition (only for agent)
                transition = Transition(
                    observation=observation,
                    actions={"play": agent_output.actions["play"]},
                    log_probs={"play": agent_output.log_probs["play"]},
                    value=agent_output.value.detach(),
                    reward=torch.tensor(reward_value, dtype=torch.float32),
                    done=torch.tensor(1.0 if done else 0.0, dtype=torch.float32),
                    masks={"play": torch.as_tensor(mask, dtype=torch.float32)},
                )
                buffer.add(transition)
                
            else:  # Bot's turn
                # Bot makes decision
                action = bot.select_action(game, 1)
                card_index = action.card_index
                if card_index < 0 or card_index >= len(state.hands[1]):
                    card_index = 0
                
                # Play card
                game.play.play_card(state, 1, card_index)
            
            # Next turn
            if state.trick_history and len(state.trick_history[-1]) == 2:
                turn = state.playing_player or turn
            else:
                turn = (turn + 1) % 2

    def _compute_reward(
        self, state: GameState, player: PlayerID, prev_trick_count: int, prev_score: int
    ) -> float:
        """Compute dense reward for current step."""
        
        reward = 0.0
        
        # Trick completion reward
        if len(state.trick_history) > prev_trick_count:
            winner = state.playing_player or 0
            if winner == player:
                if state.trick_history:
                    trick = state.trick_history[-1]
                    if len(trick) == 2:
                        trick_value = sum(card.points for _, card in trick)
                        reward += 1.0 + trick_value * 0.1
            else:
                reward -= 0.5
        
        # Hand completion reward
        if not any(state.hands[p] for p in (0, 1)):
            curr_score = state.scores.get(player, 0)
            score_gain = curr_score - prev_score
            reward += score_gain * 0.1
            
            # Contract reward
            if state.playing_player == player and state.current_bid:
                player_points = 0
                for trick in state.trick_history:
                    if len(trick) == 2:
                        winner = state.playing_player or 0
                        if winner == player:
                            player_points += sum(card.points for _, card in trick)
                
                if player_points >= state.current_bid:
                    reward += 10.0
                    margin = player_points - state.current_bid
                    reward += margin * 0.05
                else:
                    reward -= 10.0
        
        return reward

