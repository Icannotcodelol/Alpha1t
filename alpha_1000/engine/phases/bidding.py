"""Bidding phase implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..exceptions import InvalidBidError
from ..marriages import MARRIAGE_POINTS, find_marriages
from ..state import GameState, PlayerID

__all__ = ["Bid", "BiddingPhase"]


@dataclass
class Bid:
    """Represents a player's bid."""

    player: PlayerID
    value: Optional[int]
    proof_suit: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Bid(player={self.player}, value={self.value}, proof_suit={self.proof_suit})"


@dataclass
class BiddingPhase:
    """Manages the auction for a single hand."""

    bids: List[Bid] = field(default_factory=list)
    challenged: Dict[PlayerID, bool] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"BiddingPhase(bids={self.bids})"

    def start(self, state: GameState) -> None:
        """Prepare the bidding phase."""

        self.bids.clear()
        self.challenged.clear()
        state.current_bid = None
        state.playing_player = None

    def place_bid(self, state: GameState, player: PlayerID, value: Optional[int]) -> None:
        """Register a bid or pass."""

        if value is None:
            self.bids.append(Bid(player=player, value=None))
            return
        min_bid = state.rules.bidding.start_bid if state.current_bid is None else state.current_bid + state.rules.bidding.increment
        if value < min_bid or value % state.rules.bidding.increment != 0:
            msg = f"Bid {value} is invalid"
            raise InvalidBidError(msg)
        self.bids.append(Bid(player=player, value=value))
        state.current_bid = value
        state.playing_player = player

    def challenge(self, state: GameState, challenger: PlayerID, target: PlayerID) -> bool:
        """Challenge a high bid, returning True if proof succeeds."""

        current_bid = state.current_bid
        if current_bid is None or current_bid <= state.rules.bidding.proof_threshold:
            msg = "No bid requiring proof"
            raise InvalidBidError(msg)
        if state.playing_player != target:
            msg = "Target is not highest bidder"
            raise InvalidBidError(msg)
        if self.challenged.get(target):
            msg = "Player already challenged"
            raise InvalidBidError(msg)
        self.challenged[target] = True
        return self._validate_proof(state, target, current_bid)

    def _validate_proof(self, state: GameState, player: PlayerID, bid: int) -> bool:
        """Return whether the player can justify the bid."""

        marriages = find_marriages(state.hands[player])
        required = bid - state.rules.bidding.start_bid
        for suit, king, queen in marriages:
            if MARRIAGE_POINTS[suit] >= required:
                self.bids.append(Bid(player=player, value=bid, proof_suit=suit.value))
                return True
        return False
