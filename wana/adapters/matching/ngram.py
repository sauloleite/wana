from collections import defaultdict
from collections.abc import Iterable, Iterator

from wana.adapters.matching.text import shingles
from wana.domain.contamination import Hit, Level
from wana.domain.example import Example


class NgramMatcher:
    name = "ngram"

    def __init__(self, n: int = 13, *, ignore_template: tuple[str, ...] = ()) -> None:
        if n < 1:
            raise ValueError("ngram size must be positive")
        self.n, self.ignore_template = n, ignore_template

    def hits(self, train: Iterable[Example], evaluation: Iterable[Example]) -> Iterator[Hit]:
        """Return one exact shared n-gram as evidence per matching pair."""
        index: dict[str, list[int]] = defaultdict(list)
        testing = tuple(evaluation)
        for position, example in enumerate(testing):
            for gram in sorted(shingles(example.text, self.n, self.ignore_template)):
                index[gram].append(position)
        for example in train:
            evidence: dict[int, str] = {}
            for gram in sorted(shingles(example.text, self.n, self.ignore_template)):
                for position in index.get(gram, ()):
                    evidence.setdefault(position, gram)
            for position in sorted(evidence):
                yield Hit(example.id, testing[position].id, Level.EXACT, 1.0, evidence[position])
