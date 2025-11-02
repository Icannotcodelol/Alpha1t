"""Alpha-1000 package."""

from __future__ import annotations

from .engine.game import TysiacGame
from .engine.rules import load_rules

__all__ = ["TysiacGame", "load_rules"]
