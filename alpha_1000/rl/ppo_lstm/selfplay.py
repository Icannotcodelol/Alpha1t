"""Self-play utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import torch

from ...engine.game import TysiacGame
from ...engine.state import PlayerID
from ..encoding import Observation, encode_action_mask, encode_state
from ..replay import ReplayWriter
from ..rewards import RewardConfig, contract_reward, trick_reward
from .agent import AgentOutput, PpoLstmAgent
from .buffer import RolloutBuffer, Transition

__all__ = ["SelfPlayConfig", "SelfPlayWorker"]


@dataclass
class SelfPlayConfig:
    """Configuration for self-play data collection."""

    games_per_iteration: int = 4
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    store_replays: bool = False
    replay_dir: Path | None = None
    curriculum_stage: int = 0


@dataclass
class SelfPlayWorker:
    """Runs self-play games collecting experience."""

    config: SelfPlayConfig
    seed: int = 0

    def __repr__(self) -> str:  # pragma: no cover - helper
        return f"SelfPlayWorker(seed={self.seed}, stage={self.config.curriculum_stage})"

    def run(self, agent: PpoLstmAgent, buffer: RolloutBuffer) -> None:
        """Execute configured games, appending transitions to buffer."""

        rng = random.Random(self.seed)
        replay_writer = self._create_replay_writer()
        for _ in range(self.config.games_per_iteration):
            game = TysiacGame.new(seed=rng.randint(0, 10_000))
            trajectory: List[Transition] = []
            self._play_single_game(agent, game, buffer, trajectory)
            if replay_writer is not None:
                replay_writer.write(trajectory)

    def _create_replay_writer(self) -> ReplayWriter | None:
        """Return replay writer if persistence is enabled."""

        if not self.config.store_replays:
            return None
        if self.config.replay_dir is None:
            raise ValueError("Replay directory must be provided when store_replays=True")
        self.config.replay_dir.mkdir(parents=True, exist_ok=True)
        return ReplayWriter(directory=self.config.replay_dir)

    def _play_single_game(
        self,
        agent: PpoLstmAgent,
        game: TysiacGame,
        buffer: RolloutBuffer,
        trajectory: List[Transition],
    ) -> None:
        """Play a single simplified game hand under self-play."""

        game.deal()
        state = game.state
        game.bidding.start(state)
        leader = (state.dealer + 1) % 2
        start_bid = game.rules.bidding.start_bid
        game.bidding.place_bid(state, leader, start_bid)
        opponent = (leader + 1) % 2
        game.bidding.place_bid(state, opponent, None)
        state.playing_player = leader
        state.current_bid = start_bid
        game.musik.reveal(state, 0)
        self._play_tricks(agent, game, buffer, trajectory)
        game.scoring.score_hand(state, contract=start_bid)

    def _play_tricks(
        self,
        agent: PpoLstmAgent,
        game: TysiacGame,
        buffer: RolloutBuffer,
        trajectory: List[Transition],
    ) -> None:
        """Iterate through card plays capturing transitions."""

        state = game.state
        turn = state.playing_player or 0
        while any(game.state.hands[player] for player in (0, 1)):
            if not state.hands[turn]:
                turn = self._next_player(state, turn)
                continue
            legal_cards = game.play.legal_cards(state, turn)
            observation = encode_state(state, turn)
            mask, index_map = encode_action_mask(state.hands[turn], legal_cards)
            agent_output = agent.act(observation, {"play": mask})
            card_index = self._map_action(agent_output.actions["play"], index_map)
            transition = self._make_transition(observation, agent_output)
            trajectory.append(transition)
            buffer.add(transition)
            self._play_selected_card(game, turn, card_index)
            self._update_transition_rewards(state, turn, transition)
            turn = self._next_player(state, turn)

    def _play_selected_card(self, game: TysiacGame, player: PlayerID, index: int) -> None:
        """Play a card resolving trick progression."""

        if index < 0 or index >= len(game.state.hands[player]):
            index = 0
        game.play.play_card(game.state, player, index)

    def _make_transition(self, observation: Observation, agent_output: AgentOutput) -> Transition:
        """Create a transition structure from agent outputs."""

        return Transition(
            observation=observation,
            actions={"play": agent_output.actions["play"]},
            log_probs={"play": agent_output.log_probs["play"]},
            value=agent_output.value.detach(),
            reward=torch.tensor(0.0, dtype=torch.float32),
            done=torch.tensor(0.0, dtype=torch.float32),
        )

    def _update_transition_rewards(self, state, player: PlayerID, transition: Transition) -> None:
        """Populate reward and done flags for a transition."""

        transition.reward = self._compute_trick_reward(state, player)
        if not any(state.hands[p] for p in (0, 1)):
            final_reward = self._compute_final_reward(state, player)
            transition.reward = transition.reward + final_reward
            transition.done = torch.tensor(1.0, dtype=torch.float32)

    def _next_player(self, state, current: PlayerID) -> PlayerID:
        """Determine the next player considering trick resolution."""

        if not state.trick_history:
            return (current + 1) % 2
        if len(state.trick_history[-1]) == 2:
            return state.playing_player or current
        return (current + 1) % 2

    def _compute_trick_reward(self, state, player: PlayerID) -> torch.Tensor:
        """Return shaped reward when a trick completes."""

        if not state.trick_history or len(state.trick_history[-1]) != 2:
            return torch.tensor(0.0, dtype=torch.float32)
        winner = state.playing_player or 0
        won = winner == player
        value = trick_reward(won=won, config=self.config.reward_config)
        return torch.tensor(value, dtype=torch.float32)

    def _compute_final_reward(self, state, player: PlayerID) -> torch.Tensor:
        """Return end-of-hand reward from contract outcome."""

        achieved = state.scores.get(player, 0) >= state.current_bid if state.current_bid else False
        value = contract_reward(state, achieved=achieved, config=self.config.reward_config)
        return torch.tensor(value, dtype=torch.float32)

    def _map_action(self, action: torch.Tensor, index_map: Sequence[int]) -> int:
        """Translate network action index into hand position."""

        idx = int(action.item())
        if 0 <= idx < len(index_map) and index_map[idx] != -1:
            return index_map[idx]
        for pos in index_map:
            if pos != -1:
                return pos
        return 0
