from collections.abc import Iterable

from wana.domain.example import Example
from wana.domain.score import ScoredExample, ScoreResult
from wana.ports.scorer import Scorer


def score_examples(examples: Iterable[Example], scorers: Iterable[Scorer]) -> ScoreResult:
    """Apply injected scorers, rejecting duplicate identities and score names."""
    chosen = tuple(scorers)
    result: list[ScoredExample] = []
    seen: set[str] = set()
    for example in examples:
        if example.id in seen:
            raise ValueError(f"duplicate example ID: {example.id}")
        seen.add(example.id)
        scores = tuple(score for scorer in chosen for score in scorer.score(example))
        if len({s.name for s in scores}) != len(scores):
            raise ValueError("duplicate score names")
        result.append(ScoredExample(example, scores))
    return ScoreResult(tuple(result))
