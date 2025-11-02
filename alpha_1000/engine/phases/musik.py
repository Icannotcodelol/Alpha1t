"""Musik phase logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..exceptions import BombingError
from ..state import GameState, PlayerID

__all__ = ["MusikPhase"]


@dataclass
class MusikPhase:
    """Handles musik selection and bombing."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "MusikPhase()"

    def reveal(self, state: GameState, musik_index: int) -> List[str]:
        """Reveal cards from the chosen musik."""

        musiki = state.musiki[musik_index]
        return [repr(card) for card in musiki]

    def bomb(self, state: GameState, player: PlayerID, hand_index: int, reason: str) -> None:
        """Trigger a bomb if allowed."""

        if not state.rules.bombing.enabled:
            raise BombingError("Bombing disabled by rules")
        if state.bombs_remaining.get(player, 0) <= 0:
            raise BombingError("No bombs remaining")
        state.record_bomb(player=player, hand_index=hand_index, reason=reason)

    def return_cards(self, state: GameState, player: PlayerID, cards: Sequence[int]) -> None:
        """Return cards by index back to discard pile."""

        hand = state.hands[player]
        if len(cards) != 2:
            raise ValueError("Exactly two cards must be returned")
        to_discard = []
        for index in sorted(cards, reverse=True):
            if index < 0 or index >= len(hand):
                raise IndexError("Card index out of range")
            to_discard.append(hand.pop(index))
        state.discard_pile.extend(to_discard)
