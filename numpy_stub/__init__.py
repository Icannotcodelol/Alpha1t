"""Minimal numpy subset used for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]


@dataclass
class ndarray:
    """Very small ndarray implementation."""

    data: List[Union[Number, List[Number]]]

    def __post_init__(self) -> None:
        if not isinstance(self.data, list):
            self.data = [self.data]

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self.data)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, item):  # pragma: no cover - trivial
        return self.data[item]

    def __setitem__(self, key, value):  # pragma: no cover - trivial
        self.data[key] = value

    @property
    def shape(self) -> Tuple[int, ...]:
        if not self.data:
            return (0,)
        first = self.data[0]
        if isinstance(first, list):
            inner = ndarray(first).shape
            return (len(self.data),) + inner
        return (len(self.data),)

    def sum(self) -> Number:
        total = 0.0
        for value in self.data:
            if isinstance(value, list):
                total += ndarray(value).sum()
            else:
                total += float(value)
        return total


float32 = "float32"
int8 = "int8"


def array(data: Iterable[Number], dtype: str | None = None) -> ndarray:
    return ndarray(list(data))


def zeros(shape: Tuple[int, ...] | int, dtype: str | None = None) -> ndarray:
    if isinstance(shape, int):
        return ndarray([0.0 for _ in range(shape)])
    if len(shape) == 1:
        return ndarray([0.0 for _ in range(shape[0])])
    if len(shape) == 2:
        return ndarray([[0.0 for _ in range(shape[1])] for _ in range(shape[0])])
    if len(shape) == 3:
        return ndarray(
            [
                [[0.0 for _ in range(shape[2])] for _ in range(shape[1])]
                for _ in range(shape[0])
            ]
        )
    raise ValueError("Only up to 3D zeros supported")


def stack(arrays: Sequence[ndarray]) -> ndarray:
    return ndarray([list(array.data) for array in arrays])


def isclose(value: float, target: float, atol: float = 1e-8) -> bool:
    return abs(value - target) <= atol
