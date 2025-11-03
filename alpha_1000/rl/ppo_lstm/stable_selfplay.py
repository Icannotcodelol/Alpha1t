"""Improved self-play with dense rewards and better state tracking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import torch

from ...engine.game import TysiacGame
from ...engine.state import GameState, PlayerID
from ..dense_rewards import DenseRewardConfig, compute_dense_reward
from ..encoding import (
    encode_action_mask,
    encode_bid_action_mask,
    encode_bomb_action_mask,
    encode_state,
)
from .agent import PpoLstmAgent
from .buffer import RolloutBuffer, Transition
from .selfplay import SelfPlayConfig

__all__ = ["StableSelfPlayWorker"]


@dataclass
class StableSelfPlayWorker:
    """Improved self-play with dense rewards."""

    config: SelfPlayConfig
    seed: int = 0
    dense_rewards: DenseRewardConfig = None  # type: ignore[assignment]
    
    def __post_init__(self):
        if self.dense_rewards is None:
            self.dense_rewards = DenseRewardConfig()

    def run(self, agent: PpoLstmAgent, buffer: RolloutBuffer) -> None:
        """Execute games collecting experience with dense rewards."""
        
        rng = random.Random(self.seed)
        
        for game_idx in range(self.config.games_per_iteration):
            game = TysiacGame.new(seed=rng.randint(0, 100_000))
            
            if self.config.full_game:
                self._play_full_game(agent, game, buffer)
            else:
                self._play_single_hand(agent, game, buffer)

    def _play_full_game(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Play full game to 1000 points."""
        
        while not game.is_finished():
            self._play_single_hand(agent, game, buffer)

    def _play_single_hand(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Play one complete hand."""
        
        # Deal cards
        game.deal()
        state = game.state
        
        # Bidding phase - simplified for now
        game.bidding.start(state)
        leader = (state.dealer + 1) % 2
        start_bid = game.rules.bidding.start_bid
        
        # Simple bidding: leader bids start, opponent passes
        game.bidding.place_bid(state, leader, start_bid)
        game.bidding.place_bid(state, (leader + 1) % 2, None)
        state.playing_player = leader
        state.current_bid = start_bid
        
        # Musik phase (simplified - no bombing for now)
        game.musik.reveal(state, 0)
        
        # Return cards if needed
        player = state.playing_player or 0
        if len(state.hands[player]) > 10:
            # Return lowest value cards
            sorted_hand = sorted(
                enumerate(state.hands[player]), 
                key=lambda x: x[1].points
            )
            discard_indices = [sorted_hand[0][0], sorted_hand[1][0]]
            # Remove in reverse order to maintain indices
            for idx in sorted(discard_indices, reverse=True):
                if idx < len(state.hands[player]):
                    state.discard_pile.append(state.hands[player].pop(idx))
        
        # Play phase - collect transitions
        self._play_tricks(agent, game, buffer)
        
        # Score the hand
        game.scoring.score_hand(state, contract=state.current_bid or start_bid)

    def _play_tricks(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer
    ) -> None:
        """Play all tricks collecting transitions with dense rewards."""
        
        state = game.state
        turn = state.playing_player or 0
        step_count = 0
        max_steps = 100  # Safety limit
        
        while any(state.hands[p] for p in (0, 1)) and step_count < max_steps:
            step_count += 1
            
            # Skip if current player has no cards
            if not state.hands[turn]:
                turn = (turn + 1) % 2
                continue
            
            # Save previous state for reward computation
            prev_trick_count = len(state.trick_history)
            prev_score = state.scores.get(turn, 0)
            
            # Get legal actions
            legal_cards = game.play.legal_cards(state, turn)
            if not legal_cards:
                turn = (turn + 1) % 2
                continue
            
            # Encode state
            observation = encode_state(state, turn)
            mask, index_map = encode_action_mask(state.hands[turn], legal_cards)
            
            # Agent action
            agent_output = agent.act(observation, {"play": mask})
            card_action = int(agent_output.actions["play"].item())
            card_index = index_map[card_action] if 0 <= card_action < len(index_map) else 0
            
            # Ensure valid index
            if card_index < 0 or card_index >= len(state.hands[turn]):
                card_index = 0
            
            # Play card
            game.play.play_card(state, turn, card_index)
            
            # Compute dense reward
            reward_value = self._compute_step_reward(
                state, turn, prev_trick_count, prev_score
            )
            
            # Check if hand/game is done
            done = not any(state.hands[p] for p in (0, 1))
            
            # Create transition
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
            
            # Determine next player
            if state.trick_history and len(state.trick_history[-1]) == 2:
                turn = state.playing_player or turn
            else:
                turn = (turn + 1) % 2

    def _compute_step_reward(
        self, state: GameState, player: PlayerID, prev_trick_count: int, prev_score: int
    ) -> float:
        """Compute reward for current step with stronger signals."""
        
        reward = 0.0
        
        # Check if trick was just won (SCALED UP for better learning)
        if len(state.trick_history) > prev_trick_count:
            winner = state.playing_player or 0
            if winner == player:
                # Won the trick - STRONG positive signal
                if state.trick_history:
                    trick = state.trick_history[-1]
                    if len(trick) == 2:
                        trick_value = sum(card.points for _, card in trick)
                        reward += 1.0 + trick_value * 0.1  # 10x stronger
            else:
                reward -= 0.5  # 10x stronger penalty
        
        # Check if hand completed
        if not any(state.hands[p] for p in (0, 1)):
            # Hand is complete, give STRONG final reward
            curr_score = state.scores.get(player, 0)
            score_gain = curr_score - prev_score
            
            # Scale score gain significantly
            reward += score_gain * 0.1  # 10x stronger
            
            # STRONG bonus/penalty for contract outcomes
            if state.playing_player == player and state.current_bid:
                # Calculate actual points earned
                player_points = 0
                for trick in state.trick_history:
                    if len(trick) == 2:
                        winner = state.playing_player or 0
                        if winner == player:
                            player_points += sum(card.points for _, card in trick)
                
                if player_points >= state.current_bid:
                    reward += 10.0  # Made contract - BIG reward
                    # Bonus for exceeding contract
                    margin = player_points - state.current_bid
                    reward += margin * 0.05
                else:
                    reward -= 10.0  # Failed contract - BIG penalty
        
        return reward

