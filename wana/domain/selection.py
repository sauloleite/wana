"""Budget and auditable subset decisions."""

import math
from dataclasses import dataclass

from wana.domain.score import ScoredExample


@dataclass(frozen=True)
class Budget:
    keep: int | float

    def __post_init__(self) -> None:
        if isinstance(self.keep, bool) or (isinstance(self.keep, int) and self.keep < 0):
            raise ValueError("count must be nonnegative")
        if isinstance(self.keep, float) and not 0 <= self.keep <= 1:
            raise ValueError("fraction must be in [0, 1]")

    def count(self, size: int) -> int:
        return min(size, self.keep) if isinstance(self.keep, int) else math.floor(size * self.keep)


@dataclass(frozen=True)
class Decision:
    example_id: str
    decision: str
    reason: str


@dataclass(frozen=True)
class Selection:
    examples: tuple[ScoredExample, ...]
    decisions: tuple[Decision, ...]
