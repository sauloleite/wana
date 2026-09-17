from wana.domain.score import ScoredExample
from wana.domain.selection import Budget, Decision, Selection


class TopKSelector:
    def __init__(self, by: str = "ifd", *, exclude_above_one: bool = True) -> None:
        self.by, self.exclude_above_one = by, exclude_above_one

    def eligible(self, row: ScoredExample) -> bool:
        return not (
            self.exclude_above_one and any(s.name == "ifd" and s.value > 1 for s in row.scores)
        )

    def ranked(self, examples: tuple[ScoredExample, ...]) -> list[int]:
        return sorted(
            (i for i, row in enumerate(examples) if self.eligible(row)),
            key=lambda i: (-examples[i].value(self.by), i),
        )

    def result(
        self,
        examples: tuple[ScoredExample, ...],
        kept: set[int],
        reasons: dict[int, str] | None = None,
    ) -> Selection:
        return Selection(
            tuple(row for i, row in enumerate(examples) if i in kept),
            tuple(
                Decision(
                    row.example.id,
                    "KEEP" if i in kept else "DROP",
                    "selected"
                    if i in kept
                    else "ifd_above_1"
                    if not self.eligible(row)
                    else (reasons or {}).get(i, "budget"),
                )
                for i, row in enumerate(examples)
            ),
        )

    def select(self, examples: tuple[ScoredExample, ...], budget: Budget) -> Selection:
        ranked = self.ranked(examples)
        return self.result(examples, set(ranked[: budget.count(len(examples))]))
