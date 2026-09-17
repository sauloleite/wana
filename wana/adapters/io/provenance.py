"""Collect and verify content digests at the filesystem boundary."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wana.domain.manifest import Digest


def digest(path: Path, records: int | None = None) -> Digest:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return Digest(str(path), value.hexdigest(), records)


@dataclass(frozen=True)
class Verification:
    ok: bool
    checked: int
    errors: tuple[str, ...]


def verify_manifest(path: str | Path) -> Verification:
    """Verify all direct artifact hashes and the parent file; no authenticity claim."""
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list) for key in ("inputs", "eval_sets", "outputs")
    ):
        raise ValueError("manifest requires inputs, eval_sets and outputs arrays")
    entries = [item for key in ("inputs", "eval_sets", "outputs") for item in data[key]]
    if data.get("parent"):
        entries.append(data["parent"])
    errors: list[str] = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise ValueError("manifest entries require path and sha256 strings")
        target = Path(entry["path"])
        # v0.5 manifests use absolute paths; old relative paths were cwd-relative.
        if data.get("path_base") == "manifest" and not target.is_absolute():
            target = manifest_path.parent / target
        try:
            if digest(target).sha256 != entry["sha256"]:
                errors.append(f"hash mismatch: {entry['path']}")
        except OSError:
            errors.append(f"unreadable: {entry['path']}")
    return Verification(not errors, len(entries), tuple(errors))


def explain_manifest(path: str | Path) -> str:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    checked = verify_manifest(path)
    lines = [
        f"# Wana {data['wana_version']}",
        "",
        f"Verification: {'PASS' if checked.ok else 'FAIL'} ({checked.checked} files)",
        "",
    ]
    for step in data["steps"]:
        lines.append(f"- {step['name']}: " + json.dumps(step, ensure_ascii=False, sort_keys=True))
    lines.extend(f"- {error}" for error in checked.errors)
    return "\n".join(lines) + "\n"
