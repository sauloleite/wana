"""Coordinate injected matchers without accessing files or concrete adapters."""

from collections.abc import Callable, Iterable

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
    training = tuple(train)
    return check_replayed(lambda: iter(training), evaluation, matchers=matchers, fail_on=fail_on)


def check_replayed(
    train: Callable[[], Iterable[Example]],
    evaluation: Iterable[Example],
    *,
    matchers: Iterable[Matcher],
    fail_on: tuple[Level, ...] = (Level.EXACT, Level.NEAR),
) -> Report:
    """Replay training for each matcher without retaining its record bodies.

    The caller supplies the same ordered dataset on every invocation.
    Evaluation, identities and emitted evidence remain in memory.
    """
    testing = tuple(evaluation)
    if len({example.id for example in testing}) != len(testing):
        raise ValueError("duplicate example IDs in evaluation")
    seen: set[str] = set()
    for example in train():
        if example.id in seen:
            raise ValueError("duplicate example IDs in train")
        seen.add(example.id)
    count = len(seen)
    del seen
    hits = tuple(hit for matcher in matchers for hit in matcher.hits(train(), testing))
    return Report(hits, count, len(testing), fail_on)
