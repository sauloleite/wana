from typing import Protocol

from wana.domain.example import Message


class LogProbProvider(Protocol):
    def mean_nll(self, prefix: tuple[Message, ...], target: str) -> float: ...
