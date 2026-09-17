from typing import Any

from wana.domain.manifest import Digest, Manifest, Step


def build_manifest(
    *,
    version: str,
    created_at: str,
    inputs: tuple[Digest, ...],
    eval_sets: tuple[Digest, ...],
    steps: tuple[Step | dict[str, Any], ...],
    outputs: tuple[Digest, ...],
    parent: Digest | None = None,
) -> Manifest:
    """Build provenance from facts collected by the I/O boundary."""
    return Manifest(version, created_at, inputs, eval_sets, steps, outputs, parent)
