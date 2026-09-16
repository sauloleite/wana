"""Coordinate injected matchers without accessing files or concrete adapters."""

from collections.abc import Iterable

from wana.domain.contamination import Level, Report
from wana.domain.example import Example
from wana.ports.matcher import Matcher


def check_examples(
    train: Iterable[Example],
    evaluation: Iterable[Example],
    *,
    matchers: Iterable[Matcher],
    fail_on: tuple[Level, ...] = (Level.EXACT, Level.NEAR),
) -> Report:
    """Retain all evidence, including multiple detection levels for a pair."""
    training, testing = tuple(train), tuple(evaluation)
    for name, examples in (("train", training), ("evaluation", testing)):
        if len({example.id for example in examples}) != len(examples):
            raise ValueError(f"duplicate example IDs in {name}")
    hits = tuple(hit for matcher in matchers for hit in matcher.hits(training, testing))
    return Report(hits, len(training), len(testing), fail_on)
