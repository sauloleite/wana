from collections.abc import Iterator
from typing import Protocol

from wana.domain.example import Example


class ExampleReader(Protocol):
    def read(self, path: str) -> Iterator[Example]: ...
