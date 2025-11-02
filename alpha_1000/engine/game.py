"""Main game loop orchestrator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional

from .cards import Card, create_deck, sort_cards
from .phases.bidding import BiddingPhase
from .phases.musik import MusikPhase
from .phases.play import PlayPhase
from .phases.scoring import ScoringPhase
from .rules import RulesConfig, load_rules
from .state import GameState, PlayerID, create_initial_state

__all__ = ["TysiacGame"]


@dataclass
class TysiacGame:
    """High-level controller for full games."""

    rules: RulesConfig
    bidding: BiddingPhase
    musik: MusikPhase
    play: PlayPhase
    scoring: ScoringPhase
    state: GameState

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TysiacGame(scores={self.state.scores})"

    @classmethod
    def new(cls, rules: Optional[RulesConfig] = None, seed: int = 0) -> "TysiacGame":
        """Construct a new game instance with default phases."""

        cfg = rules or load_rules()
        state = create_initial_state(cfg, seed=seed)
        return cls(
            rules=cfg,
            bidding=BiddingPhase(),
            musik=MusikPhase(),
            play=PlayPhase(),
            scoring=ScoringPhase(),
            state=state,
        )

    def deal(self) -> None:
        """Deal cards to both players and populate musiki."""

        deck = list(create_deck())
        random.Random(self.state.rng_seed).shuffle(deck)
        hands = {0: sort_cards(deck[:10]), 1: sort_cards(deck[10:20])}
        self.state.set_hands(hands)
        self.state.musiki = (deck[20:22], deck[22:24])

    def run_hand(self) -> None:
        """Execute a simplified hand sequence."""

        self.deal()
        self.bidding.start(self.state)
        # Placeholder: auto bids minimal contract alternating players
        leader = (self.state.dealer + 1) % 2
        self.bidding.place_bid(self.state, leader, self.rules.bidding.start_bid)
        opponent = (leader + 1) % 2
        self.bidding.place_bid(self.state, opponent, None)
        self.state.playing_player = leader
        self.state.current_bid = self.rules.bidding.start_bid
        # Musik reveal without modifications
        self.musik.reveal(self.state, 0)
        # Simple play: players discard cards alternately
        for turn in range(10):
            player = (leader + turn) % 2
            if not self.state.hands[player]:
                continue
            self.play.play_card(self.state, player, 0)
        self.scoring.score_hand(self.state, contract=self.state.current_bid or self.rules.bidding.start_bid)

    def is_finished(self) -> bool:
        """Return True if either player reached target score."""

        return any(score >= self.rules.game.target_score for score in self.state.scores.values())
