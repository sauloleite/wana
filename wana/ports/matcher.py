from collections.abc import Iterable
from typing import Protocol

from wana.domain.contamination import Hit
from wana.domain.example import Example


class Matcher(Protocol):
    @property
    def name(self) -> str: ...
    def hits(self, train: Iterable[Example], evaluation: Iterable[Example]) -> Iterable[Hit]: ...
