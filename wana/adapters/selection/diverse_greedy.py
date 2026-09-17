from wana.adapters.embedding.hashing import HashingEmbedder
from wana.adapters.selection.topk import TopKSelector
from wana.domain.score import ScoredExample
from wana.domain.selection import Budget, Selection
from wana.domain.vectors import cosine
from wana.ports.embedder import Embedder


class DiverseGreedySelector(TopKSelector):
    def __init__(
        self,
        embedder: Embedder,
        by: str = "ifd",
        *,
        threshold: float = 0.9,
        exclude_above_one: bool = True,
    ) -> None:
        super().__init__(by, exclude_above_one=exclude_above_one)
        if not -1 <= threshold <= 1:
            raise ValueError("cosine threshold must be in [-1, 1]")
        self.embedder, self.threshold = embedder, threshold

    def select(self, examples: tuple[ScoredExample, ...], budget: Budget) -> Selection:
        if isinstance(self.embedder, HashingEmbedder):
            self.embedder.fit(row.example.text for row in examples)
        kept: set[int] = set()
        vectors: list[tuple[float, ...]] = []
        reasons: dict[int, str] = {}
        for i in self.ranked(examples):
            if len(kept) >= budget.count(len(examples)):
                break
            vector = self.embedder.embed(examples[i].example.text)
            cosine(vector, vector)
            if any(cosine(vector, previous) >= self.threshold for previous in vectors):
                reasons[i] = "similarity_threshold"
            else:
                kept.add(i)
                vectors.append(vector)
        return self.result(examples, kept, reasons)
