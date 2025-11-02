"""Game state models for Tysiąc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cards import Card
from .rules import RulesConfig

__all__ = ["PlayerID", "BombRecord", "GameState", "create_initial_state"]

PlayerID = int


@dataclass
class BombRecord:
    """Records a bombing event."""

    player: PlayerID
    hand_index: int
    reason: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"BombRecord(player={self.player}, hand_index={self.hand_index}, reason={self.reason})"


@dataclass
class GameState:
    """Complete game state snapshot."""

    rules: RulesConfig
    scores: Dict[PlayerID, int] = field(default_factory=lambda: {0: 0, 1: 0})
    dealer: PlayerID = 0
    playing_player: Optional[PlayerID] = None
    current_bid: Optional[int] = None
    hands: Dict[PlayerID, List[Card]] = field(default_factory=lambda: {0: [], 1: []})
    musiki: Tuple[List[Card], List[Card]] = field(default_factory=lambda: ([], []))
    discard_pile: List[Card] = field(default_factory=list)
    trick_history: List[List[Tuple[PlayerID, Card]]] = field(default_factory=list)
    meld_history: List[Tuple[PlayerID, Card, Card]] = field(default_factory=list)
    bombs_remaining: Dict[PlayerID, int] = field(default_factory=dict)
    bomb_events: List[BombRecord] = field(default_factory=list)
    rng_seed: int = 0

    def __repr__(self) -> str:  # pragma: no cover - simple
        return (
            "GameState("  # pylint: disable=line-too-long
            f"scores={self.scores}, dealer={self.dealer}, playing_player={self.playing_player}, "
            f"current_bid={self.current_bid})"
        )

    @property
    def trick_count(self) -> int:
        """Return number of completed tricks."""

        return len(self.trick_history)

    def set_hands(self, hands: Dict[PlayerID, Sequence[Card]]) -> None:
        """Set player hands ensuring copies are stored."""

        self.hands = {pid: list(cards) for pid, cards in hands.items()}

    def encode_scores(self) -> np.ndarray:
        """Return scores vector for neural pipelines."""

        return np.array([self.scores.get(0, 0), self.scores.get(1, 0)], dtype=np.float32)

    def record_bomb(self, player: PlayerID, hand_index: int, reason: str) -> None:
        """Append a bomb record and decrement availability."""

        if self.bombs_remaining.get(player, 0) <= 0:
            msg = f"Player {player} has no bombs remaining"
            raise ValueError(msg)
        self.bombs_remaining[player] -= 1
        self.bomb_events.append(BombRecord(player=player, hand_index=hand_index, reason=reason))


def create_initial_state(rules: RulesConfig, seed: int = 0) -> GameState:
    """Create a fresh game state with initial metadata populated."""

    bombs = {0: rules.bombing.bombs_per_player, 1: rules.bombing.bombs_per_player}
    return GameState(rules=rules, bombs_remaining=bombs, rng_seed=seed)
