from collections.abc import Iterable, Iterator

from wana.domain.contamination import Hit, Level
from wana.domain.example import Example
from wana.domain.vectors import cosine
from wana.ports.embedder import Embedder


class EmbeddingMatcher:
    name = "embedding"

    def __init__(self, embedder: Embedder, threshold: float = 0.9) -> None:
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be in [-1, 1]")
        self.embedder, self.threshold = embedder, threshold

    def hits(self, train: Iterable[Example], evaluation: Iterable[Example]) -> Iterator[Hit]:
        testing = [(e, self.embedder.embed(e.text)) for e in evaluation]
        for example in train:
            vector = self.embedder.embed(example.text)
            for other, embedded in testing:
                similarity = cosine(vector, embedded)
                if similarity >= self.threshold:
                    yield Hit(
                        example.id,
                        other.id,
                        Level.SEMANTIC,
                        similarity,
                        example.text[:160] + " ↔ " + other.text[:160],
                    )
