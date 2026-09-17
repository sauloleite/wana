"""Lossless original-record writing with namespaced, auditable annotations."""

import copy
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from wana.domain.example import Example
from wana.domain.score import Score, ScoredExample
from wana.domain.selection import Decision


def from_example(example: Example) -> ScoredExample:
    block = example.original.get("wana", {})
    if not isinstance(block, dict) or not isinstance(block.get("scores", {}), dict):
        raise ValueError("wana.scores must be an object")
    flags = block.get("flags", [])
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise ValueError("wana.flags must be a list of strings")
    scores = []
    for name, value in block.get("scores", {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"score {name} must be numeric")
        scores.append(Score(name, float(value), tuple(flags) if name == "ifd" else ()))
    return ScoredExample(example, tuple(scores))


def write_scored(
    path: Path, examples: Iterable[ScoredExample], decisions: Iterable[Decision] = ()
) -> int:
    """Atomically replace the destination only after every record is written."""
    lookup = {d.example_id: d for d in decisions}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            for row in examples:
                record = copy.deepcopy(row.example.original) or {
                    "messages": [
                        {"role": m.role.value, "content": m.content} for m in row.example.messages
                    ]
                }
                annotation = {
                    "id": row.example.id,
                    "scores": {s.name: s.value for s in row.scores},
                    "flags": sorted({flag for s in row.scores for flag in s.flags}),
                }
                if row.example.id in lookup:
                    annotation.update(asdict(lookup[row.example.id]))
                record["wana"] = annotation
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                )
                count += 1
        os.replace(temporary, path)
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
