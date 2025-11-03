"""Comprehensive state and action encodings for reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..engine.actions import mask_playable_cards
from ..engine.cards import Card, Suit
from ..engine.state import GameState, PlayerID

__all__ = [
    "MAX_HAND_CARDS",
    "CARDS_DIM",
    "MAX_TRICKS",
    "CARDS_PER_TRICK",
    "META_DIM",
    "encode_state",
    "encode_action_mask",
    "encode_bid_action_mask",
    "encode_bomb_action_mask",
    "Observation",
]

CARDS_PER_SUIT = 6
CARDS_DIM = len(Suit) * CARDS_PER_SUIT
MAX_HAND_CARDS = 12
MAX_TRICKS = 10
CARDS_PER_TRICK = 2
META_DIM = 20


@dataclass(frozen=True)
class Observation:
    """Structured numpy observation used by the PPO pipeline."""

    tensors: Dict[str, np.ndarray]

    def __repr__(self) -> str:  # pragma: no cover - trivial helper
        keys = ", ".join(sorted(self.tensors))
        return f"Observation(keys=[{keys}])"


def encode_state(state: GameState, player: PlayerID) -> Observation:
    """Encode the given state from the player's perspective."""

    hand_matrix, hand_mask = _encode_cards(state.hands[player])
    trick_matrix, trick_mask = _encode_current_trick(state)
    history_matrix, history_mask = _encode_trick_history(state)
    melds = _encode_melds(state, player)
    bombs = _encode_bombs(state, player)
    scores = _encode_scores(state, player)
    meta = _encode_meta(state, player)
    tensors = {
        "hand": hand_matrix,
        "hand_mask": hand_mask,
        "current_trick": trick_matrix,
        "current_trick_mask": trick_mask,
        "trick_history": history_matrix,
        "trick_history_mask": history_mask,
        "melds": melds,
        "bombs": bombs,
        "scores": scores,
        "meta": meta,
    }
    return Observation(tensors=tensors)


def encode_action_mask(
    hand: Sequence[Card],
    legal_cards: Iterable[Card],
    *,
    limit: int = MAX_HAND_CARDS,
) -> Tuple[np.ndarray, List[int]]:
    """Return a padded mask and card index mapping."""

    legal_mask = mask_playable_cards(hand, legal_cards).values
    padded = np.array([0.0] * limit, dtype=np.float32)
    index_map: List[int] = [-1] * limit
    for idx, flag in enumerate(legal_mask[:limit]):
        padded[idx] = float(flag)
        index_map[idx] = idx if idx < len(hand) else -1
    return padded, index_map


def encode_bid_action_mask(state: GameState) -> Tuple[np.ndarray, List[int | None]]:
    """Return a fixed-size mask for bidding options and an index→bid map.

    Index 0 corresponds to pass (value None). Indices 1..19 correspond to
    incremental bids starting from the current minimum bid.
    """

    size = 20
    mask = np.zeros(size, dtype=np.float32)
    index_map: List[int | None] = [None] * size
    mask[0] = 1.0  # pass always allowed
    index_map[0] = None
    min_bid = state.rules.bidding.start_bid if state.current_bid is None else state.current_bid + state.rules.bidding.increment
    increment = state.rules.bidding.increment
    for i in range(1, size):
        value = min_bid + (i - 1) * increment
        mask[i] = 1.0
        index_map[i] = int(value)
    return mask, index_map


def encode_bomb_action_mask(state: GameState, player: PlayerID) -> Tuple[np.ndarray, List[int]]:
    """Return a mask for bombing decision.

    Index 0: bomb, Index 1: continue. Bombing is masked out if not allowed.
    """

    allowed = state.rules.bombing.enabled and state.bombs_remaining.get(player, 0) > 0
    mask = np.array([1.0 if allowed else 0.0, 1.0], dtype=np.float32)
    return mask, [0, 1]


def _encode_cards(cards: Sequence[Card]) -> Tuple[np.ndarray, np.ndarray]:
    """Return padded card encodings and mask."""

    matrix = [[0.0] * CARDS_DIM for _ in range(MAX_HAND_CARDS)]
    mask_values = [0.0] * MAX_HAND_CARDS
    for idx, card in enumerate(cards[:MAX_HAND_CARDS]):
        matrix[idx] = _card_vector(card)
        mask_values[idx] = 1.0
    return np.array(matrix, dtype=np.float32), np.array(mask_values, dtype=np.float32)


def _encode_current_trick(state: GameState) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the active trick, padding missing cards."""

    rows = [[0.0] * CARDS_DIM for _ in range(CARDS_PER_TRICK)]
    mask = [0.0] * CARDS_PER_TRICK
    if not state.trick_history:
        return np.array(rows, dtype=np.float32), np.array(mask, dtype=np.float32)
    trick = state.trick_history[-1]
    for idx, (_, card) in enumerate(trick[:CARDS_PER_TRICK]):
        rows[idx] = _card_vector(card)
        mask[idx] = 1.0
    return np.array(rows, dtype=np.float32), np.array(mask, dtype=np.float32)


def _encode_trick_history(state: GameState) -> Tuple[np.ndarray, np.ndarray]:
    """Return encodings for completed tricks."""

    matrix = [
        [[0.0] * CARDS_DIM for _ in range(CARDS_PER_TRICK)]
        for _ in range(MAX_TRICKS)
    ]
    mask = [[0.0] * CARDS_PER_TRICK for _ in range(MAX_TRICKS)]
    completed = [trick for trick in state.trick_history if len(trick) == CARDS_PER_TRICK]
    start = max(0, len(completed) - MAX_TRICKS)
    for trick_index, trick in enumerate(completed[start:]):
        for card_index, (_, card) in enumerate(trick[:CARDS_PER_TRICK]):
            matrix[trick_index][card_index] = _card_vector(card)
            mask[trick_index][card_index] = 1.0
    return np.array(matrix, dtype=np.float32), np.array(mask, dtype=np.float32)


def _encode_melds(state: GameState, player: PlayerID) -> np.ndarray:
    """Return binary indicators for declared melds per suit."""

    matrix = [[0.0] * len(Suit) for _ in range(2)]
    for owner, king, queen in state.meld_history[-MAX_TRICKS:]:
        suit_index = list(Suit).index(king.suit)
        matrix[owner][suit_index] = 1.0
        matrix[owner][suit_index] = max(matrix[owner][suit_index], 1.0 if queen.suit == king.suit else 0.0)
    reordered = [[0.0] * len(Suit) for _ in range(2)]
    reordered[0] = matrix[player]
    opponent = 1 - player
    reordered[1] = matrix[opponent]
    return np.array(reordered, dtype=np.float32)


def _encode_bombs(state: GameState, player: PlayerID) -> np.ndarray:
    """Return bombs remaining for player and opponent."""

    opponent = 1 - player
    return np.array(
        [
            float(state.bombs_remaining.get(player, 0)),
            float(state.bombs_remaining.get(opponent, 0)),
        ],
        dtype=np.float32,
    )


def _encode_scores(state: GameState, player: PlayerID) -> np.ndarray:
    """Return normalized scores from the player's perspective."""

    target = float(state.rules.game.target_score or 1)
    opponent = 1 - player
    return np.array(
        [
            state.scores.get(player, 0) / target,
            state.scores.get(opponent, 0) / target,
        ],
        dtype=np.float32,
    )


def _encode_meta(state: GameState, player: PlayerID) -> np.ndarray:
    """Return auxiliary scalar features padded to META_DIM."""

    # Core signals
    playing = 1.0 if state.playing_player == player else 0.0
    dealer = 1.0 if state.dealer == player else 0.0
    current_bid = float(state.current_bid or 0) / 200.0
    trick_count = float(state.trick_count) / MAX_TRICKS
    bomb_count = float(len(state.bomb_events)) / max(1, MAX_TRICKS)
    trump_active = 1.0 if state.meld_history else 0.0
    musik_cards = float(sum(len(m) for m in state.musiki)) / (CARDS_PER_TRICK * 2)
    discard = float(len(state.discard_pile)) / MAX_HAND_CARDS

    # Simple phase hints: bidding, musik, play
    bidding_phase = 1.0 if state.current_bid is None and not state.trick_history else 0.0
    musik_phase = 1.0 if state.current_bid is not None and not state.trick_history else 0.0
    play_phase = 1.0 if state.trick_history else 0.0

    values = [
        playing,
        dealer,
        current_bid,
        trick_count,
        bomb_count,
        trump_active,
        musik_cards,
        discard,
        bidding_phase,
        musik_phase,
        play_phase,
    ]
    # Pad to META_DIM
    if len(values) < META_DIM:
        values.extend([0.0] * (META_DIM - len(values)))
    return np.array(values[:META_DIM], dtype=np.float32)


def _card_vector(card: Card) -> List[float]:
    """Return a plain list representation of a card encoding."""

    encoded = card.encode()
    if hasattr(encoded, "tolist"):
        return encoded.tolist()
    return [float(value) for value in encoded]
