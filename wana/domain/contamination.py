"""Evidence and policy for train/evaluation overlap."""

from dataclasses import dataclass
from enum import Enum


class Level(str, Enum):
    EXACT = "EXACT"
    NEAR = "NEAR"
    SEMANTIC = "SEMANTIC"


@dataclass(frozen=True)
class Hit:
    train_id: str
    eval_id: str
    level: Level
    value: float
    evidence: str


@dataclass(frozen=True)
class Report:
    hits: tuple[Hit, ...]
    train_records: int
    eval_records: int
    fail_on: tuple[Level, ...] = (Level.EXACT, Level.NEAR)

    @property
    def ok(self) -> bool:
        """True when no hit violates the configured policy."""
        return not any(hit.level in self.fail_on for hit in self.hits)
