"""Scoring logic for Tysiąc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

from ..cards import Card, Suit, RANK_ORDER
from ..state import GameState, PlayerID

__all__ = ["ScoringPhase", "tally_card_points"]


@dataclass
class ScoringPhase:
    """Handle end-of-hand scoring."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "ScoringPhase()"

    def score_hand(self, state: GameState, contract: int) -> Dict[PlayerID, int]:
        """Compute updated scores after a hand."""

        totals = tally_card_points(state)
        playing = state.playing_player
        if playing is None:
            return totals
        defenders = [pid for pid in totals if pid != playing]
        for defender in defenders:
            addition = ((totals[defender] + 9) // 10) * 10
            if state.rules.game.enable_lock and state.scores[defender] >= state.rules.game.lock_score:
                addition = 0
            state.scores[defender] += addition
        achieved = totals[playing] > 0 or totals[playing] >= contract
        delta = contract if achieved else -contract
        state.scores[playing] += delta
        return totals


def tally_card_points(state: GameState) -> Dict[PlayerID, int]:
    """Tally card points using trick history."""

    totals: Dict[PlayerID, int] = {0: 0, 1: 0}
    for trick in state.trick_history:
        if not trick:
            continue
        winner = _determine_winner(trick)
        points = sum(card.points for _, card in trick)
        totals[winner] += points
    return totals


def _determine_winner(trick: Sequence[Tuple[PlayerID, Card]]) -> PlayerID:
    """Determine winner mimicking play phase resolution."""

    leader, lead_card = trick[0]
    trump: Suit | None = None
    best_player = leader
    best_card = lead_card
    for player, card in trick[1:]:
        if _is_better(card, best_card, lead_card.suit, trump):
            best_player = player
            best_card = card
    return best_player


def _is_better(candidate: Card, current: Card, lead_suit: Suit, trump: Suit | None) -> bool:
    """Compare cards using rank order."""

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
