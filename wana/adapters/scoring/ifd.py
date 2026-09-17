"""Response-only loss ratio, using the final assistant turn as the target."""

import math

from wana.domain.example import Example, Role
from wana.domain.score import Score
from wana.ports.logprob import LogProbProvider


class IFDScorer:
    def __init__(self, provider: LogProbProvider) -> None:
        self.provider = provider

    def score(self, example: Example) -> tuple[Score, ...]:
        if not example.messages or example.messages[-1].role != Role.ASSISTANT:
            raise ValueError(f"IFD requires a final assistant response: {example.id}")
        target = example.messages[-1].content
        if not target.strip():
            raise ValueError(f"IFD requires a nonempty response: {example.id}")
        conditional = self.provider.mean_nll(example.messages[:-1], target)
        unconditional = self.provider.mean_nll((), target)
        if (
            not math.isfinite(conditional)
            or conditional < 0
            or not math.isfinite(unconditional)
            or unconditional <= 0
        ):
            raise ValueError(
                "IFD requires finite nonnegative conditional and positive unconditional NLL"
            )
        ratio = conditional / unconditional
        return (
            Score("ifd", ratio, ("ifd_above_1",) if ratio > 1 else ()),
            Score("conditional_nll", conditional),
            Score("unconditional_nll", unconditional),
        )
