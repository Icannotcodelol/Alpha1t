"""Trick play logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from ..cards import Card, Suit
from ..state import GameState, PlayerID

__all__ = ["PlayPhase"]


@dataclass
class PlayPhase:
    """Simple trick play manager."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "PlayPhase()"

    def legal_cards(self, state: GameState, player: PlayerID) -> List[Card]:
        """Return legal cards for the player for the current trick."""

        return list(state.hands[player])

    def play_card(self, state: GameState, player: PlayerID, card_index: int) -> None:
        """Play a card by index, starting a new trick if needed."""

        card = state.hands[player].pop(card_index)
        if not state.trick_history or len(state.trick_history[-1]) == 2:
            state.trick_history.append([])
        state.trick_history[-1].append((player, card))
        if len(state.trick_history[-1]) == 2:
            winner = self._resolve_trick(state.trick_history[-1])
            state.playing_player = winner

    def _resolve_trick(self, trick: Sequence[Tuple[PlayerID, Card]]) -> PlayerID:
        """Determine the winner of a trick."""

        leader, lead_card = trick[0]
        trump = self._current_trump(trick)
        best_player = leader
        best_card = lead_card
        for player, card in trick[1:]:
            if self._is_better(card, best_card, lead_card.suit, trump):
                best_player = player
                best_card = card
        return best_player

    def _current_trump(self, trick: Sequence[Tuple[PlayerID, Card]]) -> Suit | None:
        """Return active trump suit if any (placeholder)."""

        return None

    def _is_better(self, candidate: Card, current: Card, lead_suit: Suit, trump: Suit | None) -> bool:
        """Compare two cards to identify the winner."""

        from ..cards import RANK_ORDER

        if trump and candidate.suit == trump:
            if current.suit != trump:
                return True
            return RANK_ORDER.index(candidate.rank) > RANK_ORDER.index(current.rank)
        if current.suit == trump and candidate.suit != trump:
            return False
        if candidate.suit != lead_suit:
            return False
        if current.suit != lead_suit:
            return True
        return RANK_ORDER.index(candidate.rank) > RANK_ORDER.index(current.rank)
