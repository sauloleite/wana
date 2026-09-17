from typing import Protocol

from wana.domain.example import Example
from wana.domain.score import Score


class Scorer(Protocol):
    def score(self, example: Example) -> tuple[Score, ...]: ...
