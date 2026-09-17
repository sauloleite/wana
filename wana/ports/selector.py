from typing import Protocol

from wana.domain.score import ScoredExample
from wana.domain.selection import Budget, Selection


class Selector(Protocol):
    def select(self, examples: tuple[ScoredExample, ...], budget: Budget) -> Selection: ...
