from collections.abc import Iterable, Iterator

from wana.domain.example import Example
from wana.domain.score import ScoredExample, ScoreResult
from wana.ports.scorer import Scorer


def iter_scores(examples: Iterable[Example], scorers: Iterable[Scorer]) -> Iterator[ScoredExample]:
    """Apply injected scorers, rejecting duplicate identities and score names."""
    chosen = tuple(scorers)
    seen: set[str] = set()
    for example in examples:
        if example.id in seen:
            raise ValueError(f"duplicate example ID: {example.id}")
        seen.add(example.id)
        scores = tuple(score for scorer in chosen for score in scorer.score(example))
        if len({s.name for s in scores}) != len(scores):
            raise ValueError("duplicate score names")
        yield ScoredExample(example, scores)


def score_examples(examples: Iterable[Example], scorers: Iterable[Scorer]) -> ScoreResult:
    """Collect the streaming scorer for callers requiring a materialized result."""
    return ScoreResult(tuple(iter_scores(examples, scorers)))
