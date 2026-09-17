from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
