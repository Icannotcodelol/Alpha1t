"""Rule configuration models for Tysiąc."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

__all__ = ["GameRules", "BiddingRules", "BombingRules", "TrickRules", "RulesConfig", "load_rules", "RulesRepository", "build_default_repository"]


@dataclass
class GameRules:
    """Global game parameters."""

    target_score: int = 1000
    lock_score: int = 800
    enable_lock: bool = True

    def model_dump(self) -> Dict[str, Any]:
        return {
            "target_score": self.target_score,
            "lock_score": self.lock_score,
            "enable_lock": self.enable_lock,
        }


@dataclass
class BiddingRules:
    """Configuration of the bidding phase."""

    start_bid: int = 100
    increment: int = 10
    proof_threshold: int = 120

    def model_dump(self) -> Dict[str, Any]:
        return {
            "start_bid": self.start_bid,
            "increment": self.increment,
            "proof_threshold": self.proof_threshold,
        }


@dataclass
class BombingRules:
    """Configuration of bombing mechanics."""

    enabled: bool = True
    bombs_per_player: int = 2

    def model_dump(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "bombs_per_player": self.bombs_per_player,
        }


@dataclass
class TrickRules:
    """Rules for trick-taking interactions."""

    must_overtake: bool = True
    must_overtrump: bool = True

    def model_dump(self) -> Dict[str, Any]:
        return {
            "must_overtake": self.must_overtake,
            "must_overtrump": self.must_overtrump,
        }


@dataclass
class RulesConfig:
    """Aggregate rule model for the game."""

    game: GameRules = field(default_factory=GameRules)
    bidding: BiddingRules = field(default_factory=BiddingRules)
    bombing: BombingRules = field(default_factory=BombingRules)
    trick_rules: TrickRules = field(default_factory=TrickRules)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            "RulesConfig("  # pylint: disable=line-too-long
            f"game={self.game}, bidding={self.bidding}, bombing={self.bombing}, "
            f"trick_rules={self.trick_rules})"
        )

    def model_dump(self) -> Dict[str, Any]:
        return {
            "game": self.game.model_dump(),
            "bidding": self.bidding.model_dump(),
            "bombing": self.bombing.model_dump(),
            "trick_rules": self.trick_rules.model_dump(),
        }

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "RulesConfig":
        """Create a configuration instance from raw data."""

        def build(model_cls, values):
            if isinstance(values, dict):
                return model_cls(**values)
            return model_cls()

        return cls(
            game=build(GameRules, data.get("game", {})),
            bidding=build(BiddingRules, data.get("bidding", {})),
            bombing=build(BombingRules, data.get("bombing", {})),
            trick_rules=build(TrickRules, data.get("trick_rules", {})),
        )


DEFAULT_RULES_PATH = Path(__file__).with_name("rules_tysiac.yaml")


def load_rules(path: Path | str | None = None) -> RulesConfig:
    """Load rule configuration from YAML, falling back to defaults."""

    cfg_path = Path(path) if path is not None else DEFAULT_RULES_PATH
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = _safe_load(fh)
    return RulesConfig.model_validate(data)


def _safe_load(stream) -> Dict[str, Any]:
    try:  # pragma: no cover - optional dependency
        from yaml import safe_load

        return safe_load(stream) or {}
    except ModuleNotFoundError:  # pragma: no cover - simple fallback
        result: Dict[str, Dict[str, Any]] = {}
        current: Dict[str, Any] | None = None
        current_key: str | None = None
        for raw_line in stream.readlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not raw_line.startswith(" ") and line.endswith(":"):
                current_key = line[:-1]
                current = {}
                result[current_key] = current
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if current is None or current_key is None:
                result[key] = value
                continue
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    parsed = value
            current[key] = parsed
        return result


@dataclass
class RulesRepository:
    """Simple repository for storing and retrieving rules by name."""

    default: RulesConfig

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"RulesRepository(default={self.default})"

    def get(self, name: str | None = None) -> RulesConfig:
        """Retrieve named configuration. Only `None` is supported for now."""

        if name not in {None, "default"}:
            msg = f"Unknown rules preset: {name}"
            raise KeyError(msg)
        return self.default


def build_default_repository() -> RulesRepository:
    """Create a repository loading rules from disk."""

    return RulesRepository(default=load_rules())
