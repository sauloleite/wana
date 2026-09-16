"""Deterministic MinHash/LSH candidates, verified by exact shingle Jaccard."""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Iterator

from wana.adapters.matching.text import shingles
from wana.domain.contamination import Hit, Level
from wana.domain.example import Example


class MinHashMatcher:
    name = "minhash"

    def __init__(
        self,
        threshold: float = 0.8,
        *,
        shingle: int = 5,
        num_perm: int = 64,
        bands: int = 16,
        ignore_template: tuple[str, ...] = (),
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if shingle < 1 or num_perm < 1 or bands < 1 or num_perm % bands or num_perm > 65536:
            raise ValueError(
                "positive shingle/bands required; num_perm <= 65536 and divisible by bands"
            )
        self.threshold, self.shingle = threshold, shingle
        self.num_perm, self.bands = num_perm, bands
        self.ignore_template = ignore_template

    def _signature(self, grams: set[str]) -> tuple[int, ...]:
        return tuple(
            min(
                int.from_bytes(
                    hashlib.blake2b(
                        gram.encode(), digest_size=8, salt=perm.to_bytes(2, "big")
                    ).digest(),
                    "big",
                )
                for gram in grams
            )
            for perm in range(self.num_perm)
        )

    def _keys(self, grams: set[str]) -> Iterator[tuple[int, tuple[int, ...]]]:
        signature = self._signature(grams)
        rows = self.num_perm // self.bands
        for band in range(self.bands):
            yield band, signature[band * rows : (band + 1) * rows]

    def hits(self, train: Iterable[Example], evaluation: Iterable[Example]) -> Iterator[Hit]:
        """LSH may miss near matches; every emitted similarity is exact Jaccard."""
        testing = tuple(evaluation)
        grams = [shingles(e.text, self.shingle, self.ignore_template) for e in testing]
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        for position, group in enumerate(grams):
            if group:
                for key in self._keys(group):
                    buckets[key].append(position)
        for example in train:
            group = shingles(example.text, self.shingle, self.ignore_template)
            if not group:
                continue
            candidates = {
                position for key in self._keys(group) for position in buckets.get(key, ())
            }
            for position in sorted(candidates):
                shared = group & grams[position]
                similarity = len(shared) / len(group | grams[position])
                if similarity >= self.threshold:
                    yield Hit(example.id, testing[position].id, Level.NEAR, similarity, min(shared))
