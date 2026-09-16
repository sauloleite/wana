"""Public composition boundary: paths or already-normalized examples."""

from collections.abc import Iterable
from os import PathLike

from wana.adapters.io.jsonl import JsonlReader
from wana.adapters.matching.minhash import MinHashMatcher
from wana.adapters.matching.ngram import NgramMatcher
from wana.application.check_contamination import check_examples
from wana.domain.contamination import Level, Report
from wana.domain.example import Example
from wana.ports.matcher import Matcher

Dataset = str | PathLike[str] | Iterable[Example]


def load(source: Dataset) -> Iterable[Example]:
    return JsonlReader().read(str(source)) if isinstance(source, (str, PathLike)) else source


def check_contamination(
    train: Dataset,
    *,
    eval_sets: Iterable[Dataset],
    matchers: Iterable[Matcher] | None = None,
    fail_on: tuple[Level, ...] = (Level.EXACT, Level.NEAR),
) -> Report:
    """Check train against evaluation sets; no model, network or file writes."""
    selected = (NgramMatcher(), MinHashMatcher()) if matchers is None else matchers
    return check_examples(
        load(train),
        (example for source in eval_sets for example in load(source)),
        matchers=selected,
        fail_on=fail_on,
    )
