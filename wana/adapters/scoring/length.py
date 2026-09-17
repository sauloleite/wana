from wana.adapters.tokens.approx import ApproxTokenCounter
from wana.domain.example import Example, Role
from wana.domain.score import Score
from wana.ports.tokens import TokenCounter


class LengthScorer:
    def __init__(self, counter: TokenCounter | None = None) -> None:
        self.counter = counter or ApproxTokenCounter()

    def score(self, example: Example) -> tuple[Score, ...]:
        answer = sum(
            self.counter.count(m.content) for m in example.messages if m.role == Role.ASSISTANT
        )
        prompt = sum(
            self.counter.count(m.content) for m in example.messages if m.role != Role.ASSISTANT
        )
        return (
            Score("length", float(answer)),
            Score("response_ratio", answer / max(1, prompt)),
            Score("turns", float(sum(m.role == Role.ASSISTANT for m in example.messages))),
        )
