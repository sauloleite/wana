"""Public composition boundary: paths or already-normalized examples."""

from collections.abc import Iterable
from os import PathLike

from wana.adapters.io.jsonl import JsonlReader
from wana.adapters.matching.minhash import MinHashMatcher
from wana.adapters.matching.ngram import NgramMatcher
from wana.application.check_contamination import check_examples
from wana.domain.contamination import Level, Report
from wana.domain.example import Example
from wana.domain.score import ScoredExample, ScoreResult
from wana.domain.selection import Selection
from wana.ports.matcher import Matcher
from wana.ports.scorer import Scorer
from wana.ports.selector import Selector

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


def score_dataset(source: Dataset, *, scorers: Iterable["Scorer"] | None = None) -> "ScoreResult":
    """Score a path or loaded examples, using the pip-installed model by default."""
    from wana.adapters.logprob.bundled import BundledLogProbProvider
    from wana.adapters.scoring.ifd import IFDScorer
    from wana.adapters.scoring.length import LengthScorer
    from wana.application.score_dataset import score_examples

    chosen = (LengthScorer(), IFDScorer(BundledLogProbProvider())) if scorers is None else scorers
    return score_examples(load(source), chosen)


def select_subset(
    source: "str | PathLike[str] | ScoreResult | Iterable[ScoredExample]",
    *,
    keep: int | float = 0.2,
    by: str = "ifd",
    selector: "Selector | None" = None,
) -> "Selection":
    """Return an ordered subset and a KEEP/DROP reason for every input record."""
    from wana.adapters.io.scored import from_example
    from wana.adapters.selection.topk import TopKSelector
    from wana.application.select_subset import select_examples
    from wana.domain.score import ScoreResult
    from wana.domain.selection import Budget

    if isinstance(source, (str, PathLike)):
        rows = tuple(from_example(e) for e in load(source))
    elif isinstance(source, ScoreResult):
        rows = source.examples
    else:
        rows = tuple(source)
    return select_examples(rows, selector=selector or TopKSelector(by), budget=Budget(keep))
