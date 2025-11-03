"""Self-play utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import torch

from ...engine.game import TysiacGame
from ...engine.state import PlayerID
from ..encoding import (
    Observation,
    encode_action_mask,
    encode_bid_action_mask,
    encode_bomb_action_mask,
    encode_state,
)
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
    full_game: bool = False


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
            if self.config.full_game:
                self._play_full_game(agent, game, buffer, trajectory)
            else:
                self._play_single_hand(agent, game, buffer, trajectory)
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

    def _play_full_game(
        self,
        agent: PpoLstmAgent,
        game: TysiacGame,
        buffer: RolloutBuffer,
        trajectory: List[Transition],
    ) -> None:
        """Play a full game until target score reached."""

        while not game.is_finished():
            self._play_single_hand(agent, game, buffer, trajectory)

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
            transition = self._make_transition(observation, agent_output, masks={"play": torch.as_tensor(mask)})
            trajectory.append(transition)
            buffer.add(transition)
            self._play_selected_card(game, turn, card_index)
            self._update_transition_rewards(state, turn, transition)
            turn = self._next_player(state, turn)

    def _run_bidding(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer, trajectory: List[Transition]
    ) -> None:
        """Conduct a basic two-player bidding where both sides use the agent policy."""

        state = game.state
        leader = (state.dealer + 1) % 2
        current = leader
        passes = {0: False, 1: False}
        # Limit rounds to avoid infinite loops.
        for _ in range(10):
            observation = encode_state(state, current)
            bid_mask, index_map = encode_bid_action_mask(state)
            output = agent.act(observation, {"bid": bid_mask})
            choice = int(output.actions["bid"].item())
            bid_value = index_map[choice]
            transition = self._make_transition(
                observation,
                output,
                masks={"bid": torch.as_tensor(bid_mask)},
            )
            trajectory.append(transition)
            buffer.add(transition)
            if bid_value is None:
                passes[current] = True
            else:
                game.bidding.place_bid(state, current, bid_value)
                passes[current] = False
            other = (current + 1) % 2
            if passes[current] and passes[other]:
                break
            current = other

        if state.playing_player is None:
            # Ensure there is a declarer
            state.playing_player = leader
            state.current_bid = game.rules.bidding.start_bid

        # Auto challenge if threshold exceeded
        try:
            other = (state.playing_player + 1) % 2  # type: ignore[operator]
            if state.current_bid and state.current_bid > game.rules.bidding.proof_threshold:
                game.bidding.challenge(state, challenger=other, target=state.playing_player)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive
            pass

    def _maybe_bomb(
        self, agent: PpoLstmAgent, game: TysiacGame, buffer: RolloutBuffer, trajectory: List[Transition]
    ) -> None:
        """Let the declarer decide whether to bomb."""

        state = game.state
        player = state.playing_player or 0
        observation = encode_state(state, player)
        mask, _ = encode_bomb_action_mask(state, player)
        output = agent.act(observation, {"bomb": mask})
        transition = self._make_transition(observation, output, masks={"bomb": torch.as_tensor(mask)})
        trajectory.append(transition)
        buffer.add(transition)
        action = int(output.actions["bomb"].item())
        if action == 0 and mask[0] > 0:
            # Bomb selected
            try:
                game.musik.bomb(state, player=player, hand_index=0, reason="policy")
            except Exception:  # pragma: no cover - robust to rule errors
                pass

    def _return_discard_if_needed(self, game: TysiacGame) -> None:
        """If hand sizes exceed rules expectation after musik, return two cards."""

        state = game.state
        player = state.playing_player or 0
        if len(state.hands[player]) > 10:
            # Return first two cards by index
            try:
                game.musik.return_cards(state, player=player, cards=[0, 1])
            except Exception:  # pragma: no cover - engine leniency
                pass

    def _declare_meld_if_any(self, game: TysiacGame) -> None:
        """Record a single available marriage for the declarer if present."""

        from ...engine.marriages import find_marriages

        state = game.state
        player = state.playing_player or 0
        marriages = find_marriages(state.hands[player])
        if marriages:
            suit, king, queen = marriages[0]
            state.meld_history.append((player, king, queen))

    def _play_selected_card(self, game: TysiacGame, player: PlayerID, index: int) -> None:
        """Play a card resolving trick progression."""

        if index < 0 or index >= len(game.state.hands[player]):
            index = 0
        game.play.play_card(game.state, player, index)

    def _make_transition(self, observation: Observation, agent_output: AgentOutput, *, masks: dict[str, torch.Tensor] | None = None) -> Transition:
        """Create a transition structure from agent outputs."""

        heads = set(masks.keys()) if masks is not None else {"play"}
        return Transition(
            observation=observation,
            actions={k: v for k, v in agent_output.actions.items() if k in heads},
            log_probs={k: v for k, v in agent_output.log_probs.items() if k in heads},
            value=agent_output.value.detach(),
            reward=torch.tensor(0.0, dtype=torch.float32),
            done=torch.tensor(0.0, dtype=torch.float32),
            masks=masks if masks is not None else {},
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
    def _play_single_hand(
        self,
        agent: PpoLstmAgent,
        game: TysiacGame,
        buffer: RolloutBuffer,
        trajectory: List[Transition],
    ) -> None:
        """Play one hand including bidding, musik/bomb, melds, and tricks."""

        game.deal()
        state = game.state
        game.bidding.start(state)
        self._run_bidding(agent, game, buffer, trajectory)
        # Reveal first musik for now.
        game.musik.reveal(state, 0)
        # Optional bombing decision by the declarer
        self._maybe_bomb(agent, game, buffer, trajectory)
        # Simple discard: return two first cards if too many
        self._return_discard_if_needed(game)
        # Optional meld declaration
        self._declare_meld_if_any(game)
        # Play tricks to completion
        self._play_tricks(agent, game, buffer, trajectory)
        game.scoring.score_hand(state, contract=state.current_bid or game.rules.bidding.start_bid)
