"""Recognize a verified retry of a standalone selection at the filesystem boundary."""

import json
from pathlib import Path
from typing import Any

from wana.adapters.io.provenance import verify_manifest


def previous_selection_count(
    source: Path, manifest: Path | None, parameters: dict[str, Any], version: str
) -> int | None:
    """Return the saved count only for the same request and intact output artifact."""
    if manifest is None:
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("wana_version") != version:
        return None
    steps = data.get("steps", [])
    if not steps or not isinstance(steps[-1], dict):
        return None
    step = steps[-1]
    if step.get("name") != "select" or any(step.get(k) != v for k, v in parameters.items()):
        return None
    entries = [
        entry
        for entry in data.get("outputs", [])
        if isinstance(entry, dict) and entry.get("path") == str(source.resolve())
    ]
    if len(entries) != 1:
        return None
    count = entries[0].get("records")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    checked = verify_manifest(manifest)
    if not checked.ok:
        raise ValueError("previous selection cannot be reused: " + "; ".join(checked.errors))
    return count
