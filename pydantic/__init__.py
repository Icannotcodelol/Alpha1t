"""Minimal stub of Pydantic for testing."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any, Dict, Generic, Optional, Type, TypeVar

T = TypeVar("T")


def Field(default: Any = None, default_factory: Any | None = None, **_: Any) -> Any:
    return default if default_factory is None else default_factory()


class BaseModel:
    """Very small subset of the Pydantic API."""

    def __init__(self, **data: Any) -> None:
        for name, value in data.items():
            setattr(self, name, value)

    @classmethod
    def model_validate(cls: Type["BaseModel"], data: Dict[str, Any]) -> "BaseModel":
        return cls(**data)

    def model_dump(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}({self.__dict__})"
