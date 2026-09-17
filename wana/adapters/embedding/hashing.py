"""Stable hashed TF-IDF vectors fitted without model weights or dependencies."""

import hashlib
import math
from collections import Counter
from collections.abc import Iterable

from wana.adapters.matching.text import words


class HashingEmbedder:
    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.idf = [1.0] * dimensions

    def _bin(self, word: str) -> int:
        return int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big") % self.dimensions

    def fit(self, texts: Iterable[str]) -> None:
        """Fit smoothed inverse document frequency for each hash bin."""
        frequencies: Counter[int] = Counter()
        count = 0
        for text in texts:
            frequencies.update({self._bin(word) for word in words(text)})
            count += 1
        self.idf = [
            1 + math.log((1 + count) / (1 + frequencies[i])) for i in range(self.dimensions)
        ]

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for word, count in sorted(Counter(words(text)).items()):
            slot = self._bin(word)
            vector[slot] += (1 + math.log(count)) * self.idf[slot]
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm if norm else 0.0 for value in vector)
