"""Portable provenance records; timestamps are supplied by the caller."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Digest:
    path: str
    sha256: str
    records: int | None = None


@dataclass(frozen=True)
class Step:
    name: str
    matchers: tuple[str, ...]
    ngram: int
    threshold: float
    shingle: int
    num_perm: int
    bands: int
    ignore_template: tuple[str, ...]
    fail_on: tuple[str, ...]
    hits: int


@dataclass(frozen=True)
class Manifest:
    wana_version: str
    created_at: str
    inputs: tuple[Digest, ...]
    eval_sets: tuple[Digest, ...]
    steps: tuple[Step | dict[str, Any], ...]
    outputs: tuple[Digest, ...]
    parent: Digest | None = None
