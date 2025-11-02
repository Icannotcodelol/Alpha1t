"""Tests covering the reinforcement-learning pipeline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ...engine.game import TysiacGame
from ...rl.encoding import encode_state
from ...rl.ppo_lstm.agent import PpoLstmAgent
from ...rl.ppo_lstm.buffer import RolloutBuffer, Transition
from ...rl.ppo_lstm.selfplay import SelfPlayConfig, SelfPlayWorker
from ...rl.ppo_lstm.trainer import CurriculumManager, Trainer


def _dummy_transition() -> Transition:
    game = TysiacGame.new()
    game.deal()
    observation = encode_state(game.state, player=0)
    action = torch.tensor(0)
    log_prob = torch.tensor(0.0)
    value = torch.tensor(0.0)
    reward = torch.tensor(1.0)
    done = torch.tensor(1.0)
    return Transition(
        observation=observation,
        actions={"play": action},
        log_probs={"play": log_prob},
        value=value,
        reward=reward,
        done=done,
    )


def test_rollout_buffer_advantage_calculation() -> None:
    buffer = RolloutBuffer()
    transition = _dummy_transition()
    buffer.add(transition)
    buffer.compute_returns_and_advantages(torch.zeros(1), gamma=0.99, gae_lambda=0.95)
    assert transition.advantage is not None
    assert transition.return_ is not None


def test_selfplay_worker_collects_transitions() -> None:
    agent = PpoLstmAgent.create()
    buffer = RolloutBuffer()
    worker = SelfPlayWorker(config=SelfPlayConfig(games_per_iteration=1))
    worker.run(agent, buffer)
    assert buffer.transitions, "Expected transitions from self-play run"


def test_trainer_iteration_runs_without_error() -> None:
    agent = PpoLstmAgent.create()
    buffer = RolloutBuffer()
    curriculum = CurriculumManager(stages={0: SelfPlayConfig(games_per_iteration=1)})
    trainer = Trainer(agent=agent, buffer=buffer, curriculum=curriculum, batch_size=2, epochs=1)
    trainer.run_iteration(iteration=0)
    assert buffer.transitions, "Buffer should retain collected transitions"
