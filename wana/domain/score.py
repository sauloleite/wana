"""Validated scores attached to an immutable example."""

import math
from dataclasses import dataclass

from wana.domain.example import Example


@dataclass(frozen=True)
class Score:
    name: str
    value: float
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class ScoredExample:
    example: Example
    scores: tuple[Score, ...]

    def value(self, name: str) -> float:
        for score in self.scores:
            if score.name == name:
                return score.value
        raise ValueError(f"missing score {name!r} for {self.example.id}")


@dataclass(frozen=True)
class ScoreResult:
    examples: tuple[ScoredExample, ...]
