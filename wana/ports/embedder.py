from typing import Protocol


class Embedder(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...
