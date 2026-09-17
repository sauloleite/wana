from wana.domain.score import ScoredExample
from wana.domain.selection import Budget, Selection
from wana.ports.selector import Selector


def select_examples(
    examples: tuple[ScoredExample, ...], *, selector: Selector, budget: Budget
) -> Selection:
    if len({row.example.id for row in examples}) != len(examples):
        raise ValueError("duplicate example IDs")
    return selector.select(examples, budget)
