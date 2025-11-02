"""Exception hierarchy for Alpha-1000."""

from __future__ import annotations

__all__ = ["TysiacError", "InvalidActionError", "InvalidBidError", "BombingError"]


class TysiacError(Exception):
    """Base exception for the project."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}({self.args})"


class InvalidActionError(TysiacError):
    """Raised when an invalid move is attempted."""


class InvalidBidError(InvalidActionError):
    """Raised when a bid is not allowed by the rules."""


class BombingError(TysiacError):
    """Raised when bombing is not possible."""
