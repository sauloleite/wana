"""Compose score, select and check without filesystem or concrete adapters."""

from dataclasses import dataclass

from wana.application.check_contamination import check_examples
from wana.application.score_dataset import score_examples
from wana.application.select_subset import select_examples
from wana.domain.contamination import Report
from wana.domain.example import Example
from wana.domain.score import ScoreResult
from wana.domain.selection import Budget, Selection
from wana.ports.matcher import Matcher
from wana.ports.scorer import Scorer
from wana.ports.selector import Selector


@dataclass(frozen=True)
class PipelineResult:
    scored: ScoreResult
    selection: Selection
    contamination: Report


def process(
    examples: tuple[Example, ...],
    evaluation: tuple[Example, ...],
    *,
    scorers: tuple[Scorer, ...],
    selector: Selector,
    matchers: tuple[Matcher, ...],
    budget: Budget,
) -> PipelineResult:
    scored = score_examples(examples, scorers)
    selection = select_examples(scored.examples, selector=selector, budget=budget)
    report = check_examples((r.example for r in selection.examples), evaluation, matchers=matchers)
    return PipelineResult(scored, selection, report)
