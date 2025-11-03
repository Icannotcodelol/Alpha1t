"""Reward shaping strategies."""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.state import GameState

__all__ = [
    "RewardConfig",
    "trick_reward",
    "contract_reward",
    "meld_reward",
    "bomb_reward",
]


@dataclass
class RewardConfig:
    """Configuration values used for reward shaping."""

    trick_win_bonus: float = 0.01  # Smaller, more frequent rewards
    trick_loss_penalty: float = -0.005
    meld_bonus: float = 0.02
    bomb_penalty: float = -0.1
    precise_contract_bonus: float = 0.1
    point_scale: float = 100.0
    contract_scale: float = 0.01  # Scale contract rewards appropriately

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return (
            "RewardConfig("  # pylint: disable=line-too-long
            f"trick_win_bonus={self.trick_win_bonus}, trick_loss_penalty={self.trick_loss_penalty}, "
            f"meld_bonus={self.meld_bonus}, bomb_penalty={self.bomb_penalty}, "
            f"precise_contract_bonus={self.precise_contract_bonus}, point_scale={self.point_scale})"
        )


def trick_reward(*, won: bool, config: RewardConfig | None = None) -> float:
    """Return reward based on trick outcome."""

    cfg = config or RewardConfig()
    return cfg.trick_win_bonus if won else cfg.trick_loss_penalty


def contract_reward(state: GameState, *, achieved: bool, config: RewardConfig | None = None) -> float:
    """Return reward based on contract success or failure."""

    cfg = config or RewardConfig()
    base = float(state.current_bid or 0) * cfg.contract_scale
    if achieved:
        return base + cfg.precise_contract_bonus
    else:
        return -(base * 0.5) + cfg.trick_loss_penalty  # Negative reward for failure


def meld_reward(config: RewardConfig | None = None) -> float:
    """Return bonus for successful meld declaration."""

    cfg = config or RewardConfig()
    return cfg.meld_bonus


def bomb_reward(successful: bool, config: RewardConfig | None = None) -> float:
    """Return penalty or neutral outcome for bombing decisions."""

    cfg = config or RewardConfig()
    return 0.0 if successful else cfg.bomb_penalty
