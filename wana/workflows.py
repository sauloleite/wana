"""Filesystem workflows assembling the pure application services."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

from wana.adapters.io.provenance import digest
from wana.adapters.io.scored import write_scored
from wana.adapters.report.json import serialize
from wana.adapters.report.markdown import render_markdown
from wana.api import Dataset, load
from wana.application.build_manifest import build_manifest
from wana.application.pipeline import PipelineResult, process
from wana.domain.manifest import Manifest
from wana.domain.selection import Budget
from wana.ports.matcher import Matcher
from wana.ports.scorer import Scorer
from wana.ports.selector import Selector


@dataclass(frozen=True)
class RunResult:
    pipeline: PipelineResult
    manifest: Manifest | None


def record_manifest(
    paths: tuple[Path, ...],
    evaluation: tuple[Path, ...],
    outputs: tuple[tuple[Path, int | None], ...],
    steps: tuple[dict[str, Any], ...],
    destination: Path,
    parent: Path | None = None,
) -> Manifest:
    from wana import __version__
    from wana.adapters.io.jsonl import JsonlReader

    reader = JsonlReader()

    def entry(path: Path) -> Any:
        return digest(path.resolve(), sum(1 for _ in reader.read(str(path))))

    manifest = build_manifest(
        version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs=tuple(entry(p) for p in paths),
        eval_sets=tuple(entry(p) for p in evaluation),
        outputs=tuple(digest(p.resolve(), count) for p, count in outputs),
        steps=steps,
        parent=digest(parent.resolve()) if parent else None,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialize(manifest), encoding="utf-8", newline="\n")
    return manifest


def run(
    source: Dataset,
    *,
    out_dir: str | Path | None = None,
    eval_sets: Iterable[Dataset] = (),
    keep: int | float = 0.2,
    seed: int = 42,
    scorers: Iterable[Scorer] | None = None,
    selector: Selector | None = None,
    matchers: Iterable[Matcher] | None = None,
    parameters: dict[str, Any] | None = None,
) -> RunResult:
    """Score/select/check, optionally writing all artifacts and parent provenance.

    Path inputs use absolute artifact paths in the manifest. In-memory input hashes
    are not invented: outputs are written but no file manifest is produced.
    """
    from wana.adapters.logprob.bundled import BundledLogProbProvider
    from wana.adapters.matching.minhash import MinHashMatcher
    from wana.adapters.matching.ngram import NgramMatcher
    from wana.adapters.scoring.ifd import IFDScorer
    from wana.adapters.scoring.length import LengthScorer
    from wana.adapters.selection.topk import TopKSelector

    if isinstance(source, (str, PathLike)) and Path(source).is_dir():
        source = Path(source) / "train.jsonl"
    evaluations = tuple(eval_sets)
    source_path = Path(source).resolve() if isinstance(source, (str, PathLike)) else None
    eval_paths = tuple(Path(e).resolve() for e in evaluations if isinstance(e, (str, PathLike)))
    parent = source_path.parent / (source_path.stem + ".manifest.json") if source_path else None
    if source_path and (parent is None or not parent.is_file()):
        parent = source_path.parent / "manifest.json"
    parent = parent if parent and parent.is_file() else None
    target = Path(out_dir).resolve() if out_dir is not None else None
    if target:
        protected = {*eval_paths, source_path, parent}
        if any(
            (target / name).resolve() in protected
            for name in (
                "scored.jsonl",
                "selected.jsonl",
                "contamination.json",
                "manifest.json",
                "report.md",
            )
        ):
            raise ValueError("output would overwrite input or parent")
    provider = None
    if scorers is None:
        provider = BundledLogProbProvider()
        chosen: tuple[Scorer, ...] = (LengthScorer(), IFDScorer(provider))
    else:
        chosen = tuple(scorers)
    chosen_selector = selector or TopKSelector()
    chosen_matchers = (
        tuple(matchers) if matchers is not None else (NgramMatcher(), MinHashMatcher())
    )
    examples = tuple(load(source))
    result = process(
        examples,
        tuple(e for dataset in evaluations for e in load(dataset)),
        scorers=chosen,
        selector=chosen_selector,
        matchers=chosen_matchers,
        budget=Budget(keep),
    )
    manifest = None
    if target:
        target.mkdir(parents=True, exist_ok=True)
        scored_path, selected_path = target / "scored.jsonl", target / "selected.jsonl"
        write_scored(scored_path, result.scored.examples, result.selection.decisions)
        write_scored(selected_path, result.selection.examples, result.selection.decisions)
        report_path = target / "contamination.json"
        report_path.write_text(serialize(result.contamination), encoding="utf-8", newline="\n")
        markdown_path = target / "report.md"
        markdown_path.write_text(
            render_markdown(result.contamination), encoding="utf-8", newline="\n"
        )
        if source_path and len(eval_paths) == len(evaluations):
            prompt_hash = hashlib.sha256(
                json.dumps(
                    [[asdict(m) for m in e.messages[:-1]] for e in examples], sort_keys=True
                ).encode()
            ).hexdigest()
            model_info = (
                {"model": provider.model_name, "revision": provider.revision} if provider else {}
            )
            steps = (
                {
                    "name": "score",
                    "scorers": [type(s).__name__ for s in chosen],
                    "prompt_sha256": prompt_hash,
                    **model_info,
                    **(parameters or {}),
                },
                {
                    "name": "select",
                    "selector": type(chosen_selector).__name__,
                    "keep": keep,
                    "seed": seed,
                    "tie_break": "input_order",
                    "in": len(examples),
                    "out": len(result.selection.examples),
                },
                {
                    "name": "check",
                    "matchers": [m.name for m in chosen_matchers],
                    "hits": len(result.contamination.hits),
                    "scope": "selected",
                },
            )
            manifest = record_manifest(
                (source_path,),
                eval_paths,
                (
                    (scored_path, len(examples)),
                    (selected_path, len(result.selection.examples)),
                    (report_path, len(result.contamination.hits)),
                    (markdown_path, None),
                ),
                steps,
                target / "manifest.json",
                parent,
            )
    return RunResult(result, manifest)
