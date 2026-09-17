"""Largest-remainder allocation preserves eligible strata proportions."""

from collections import defaultdict

from wana.adapters.selection.topk import TopKSelector
from wana.domain.score import ScoredExample
from wana.domain.selection import Budget, Selection


class StratifiedSelector(TopKSelector):
    def __init__(
        self, by: str = "ifd", *, strata: str = "source", exclude_above_one: bool = True
    ) -> None:
        super().__init__(by, exclude_above_one=exclude_above_one)
        self.strata = strata

    def select(self, examples: tuple[ScoredExample, ...], budget: Budget) -> Selection:
        groups: dict[str, list[int]] = defaultdict(list)
        for i in self.ranked(examples):
            row = examples[i]
            key = (
                str(row.value("turns"))
                if self.strata == "turns"
                else row.example.metadata.get(self.strata)
            )
            if key is None:
                raise ValueError(f"missing stratum {self.strata!r} for {row.example.id}")
            groups[str(key)].append(i)
        total = sum(map(len, groups.values()))
        count = min(total, budget.count(len(examples)))
        if not total:
            return self.result(examples, set())
        quotas = {key: count * len(rows) // total for key, rows in groups.items()}
        order = sorted(groups, key=lambda key: (-(count * len(groups[key]) % total), key))
        for key in order[: count - sum(quotas.values())]:
            quotas[key] += 1
        return self.result(
            examples, {i for key, rows in groups.items() for i in rows[: quotas[key]]}
        )
