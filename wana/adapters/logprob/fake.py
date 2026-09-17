from wana.domain.example import Message


class FakeLogProbProvider:
    """Fixed losses for tests, never a model-quality estimate."""

    def __init__(self, conditional: float = 1.0, unconditional: float = 2.0) -> None:
        self.conditional, self.unconditional = conditional, unconditional

    def mean_nll(self, prefix: tuple[Message, ...], target: str) -> float:
        return self.conditional if prefix else self.unconditional
