"""CLI for scoring, selection and pipeline orchestration."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wana.adapters.embedding.hashing import HashingEmbedder
from wana.adapters.embedding.onnx import OnnxEmbedder
from wana.adapters.io.provenance import digest, explain_manifest, verify_manifest
from wana.adapters.io.scored import write_scored
from wana.adapters.logprob.bundled import BundledLogProbProvider
from wana.adapters.logprob.cache import CachedLogProbProvider
from wana.adapters.logprob.fake import FakeLogProbProvider
from wana.adapters.logprob.transformers_cpu import TransformersCPU
from wana.adapters.scoring.ifd import IFDScorer
from wana.adapters.scoring.length import LengthScorer
from wana.adapters.selection.diverse_greedy import DiverseGreedySelector
from wana.adapters.selection.stratified import StratifiedSelector
from wana.adapters.selection.topk import TopKSelector
from wana.api import score_dataset, select_subset
from wana.ports.embedder import Embedder
from wana.ports.logprob import LogProbProvider
from wana.ports.scorer import Scorer
from wana.ports.selector import Selector
from wana.workflows import record_manifest, run


def keep_value(text: str) -> int | float:
    try:
        return float(text) if "." in text or "e" in text.lower() else int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("keep must be a count or fraction") from exc


def add_commands(commands: Any) -> None:
    for name in ("score", "select", "run"):
        parser = commands.add_parser(name)
        parser.add_argument("source", type=Path)
        parser.add_argument("-o", "--output", type=Path, required=True)
        if name in ("score", "run"):
            parser.add_argument("--scorer", choices=["ifd", "length"], default="ifd")
            parser.add_argument(
                "--provider",
                choices=["bundled", "transformers_cpu", "fake"],
                default=os.environ.get("WANA_PROVIDER", "bundled"),
            )
            parser.add_argument(
                "--model",
                default=os.environ.get("WANA_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
            )
            parser.add_argument("--revision")
            parser.add_argument("--max-tokens", type=int, default=2048)
            parser.add_argument("--resume", action="store_true")
            parser.add_argument(
                "--cache-dir",
                type=Path,
                default=Path(os.environ.get("WANA_CACHE_DIR", str(Path.home() / ".cache/wana"))),
            )
        if name in ("select", "run"):
            parser.add_argument("--keep", type=keep_value, default=0.2)
            parser.add_argument("--by", default="ifd", help="score used for ranking")
            parser.add_argument(
                "--selector", choices=["topk", "stratified", "diverse_greedy"], default="topk"
            )
            parser.add_argument("--strata", default="source")
            parser.add_argument("--diverse", action="store_true")
            parser.add_argument(
                "--threshold", type=float, default=0.9, help="maximum cosine similarity"
            )
            parser.add_argument("--embedder", choices=["hashing", "onnx"], default="hashing")
            parser.add_argument("--embedding-model")
            parser.add_argument("--tokenizer")
            parser.add_argument("--include-ifd-above-one", action="store_true")
            parser.add_argument("--seed", type=int, default=42)
        if name == "run":
            parser.add_argument("--eval", dest="evaluation", type=Path, action="append", default=[])
    explain = commands.add_parser("explain")
    explain.add_argument("manifest", type=Path)
    explain.add_argument("--json", action="store_true")


def embedding(args: argparse.Namespace) -> Embedder:
    if args.embedder == "hashing":
        return HashingEmbedder()
    if not args.embedding_model or not args.tokenizer:
        raise ValueError("ONNX requires --embedding-model and --tokenizer local paths")
    return OnnxEmbedder(args.embedding_model, args.tokenizer)


def scoring(
    args: argparse.Namespace,
) -> tuple[tuple[Scorer, ...], dict[str, Any], CachedLogProbProvider | None]:
    if args.scorer == "length":
        return (LengthScorer(),), {"scorer": "length", "token_counter": "approx"}, None
    provider: LogProbProvider
    if args.provider == "bundled":
        provider = BundledLogProbProvider(max_tokens=args.max_tokens)
    elif args.provider == "fake":
        provider = FakeLogProbProvider()
    else:
        provider = TransformersCPU(args.model, revision=args.revision, max_tokens=args.max_tokens)
    info = {
        "implementation": "wana-0.5.0",
        "runtime": importlib.metadata.version("llama-cpp-python")
        if args.provider == "bundled"
        else importlib.metadata.version("transformers")
        if args.provider == "transformers_cpu"
        else "fake-v1",
        "scorer": "ifd",
        "provider": args.provider,
        "model": getattr(provider, "model_name", "fake"),
        "revision": getattr(provider, "revision", None),
        "max_tokens": args.max_tokens,
        "prompt_template_sha256": hashlib.sha256(
            str(getattr(provider, "chat_template", "")).encode()
        ).hexdigest(),
    }
    cache = None
    if args.resume:
        cache = CachedLogProbProvider(
            provider, args.cache_dir / "nll.sqlite3", json.dumps(info, sort_keys=True)
        )
        provider = cache
    return (LengthScorer(), IFDScorer(provider)), info, cache


def selecting(args: argparse.Namespace) -> Selector:
    options = {"exclude_above_one": not args.include_ifd_above_one}
    if args.diverse or args.selector == "diverse_greedy":
        return DiverseGreedySelector(embedding(args), args.by, threshold=args.threshold, **options)
    if args.selector == "stratified":
        return StratifiedSelector(args.by, strata=args.strata, **options)
    return TopKSelector(args.by, **options)


def selection_parameters(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "selector": "diverse_greedy" if args.diverse else args.selector,
        "by": args.by,
        "keep": args.keep,
        "strata": args.strata,
        "threshold": args.threshold,
        "embedder": args.embedder,
        "exclude_ifd_above_one": not args.include_ifd_above_one,
        "seed": args.seed,
        "tie_break": "input_order",
    }
    if args.embedder == "onnx" and (args.diverse or args.selector == "diverse_greedy"):
        info["embedding_model"] = asdict(digest(Path(args.embedding_model).resolve()))
        info["tokenizer"] = asdict(digest(Path(args.tokenizer).resolve()))
    return info


def execute(args: argparse.Namespace) -> int:
    if args.command == "explain":
        verification = verify_manifest(args.manifest)
        print(
            json.dumps(asdict(verification), sort_keys=True)
            if args.json
            else explain_manifest(args.manifest),
            end="\n",
        )
        return 0 if verification.ok else 1
    source = args.source / "train.jsonl" if args.source.is_dir() else args.source
    parent = source.parent / (source.stem + ".manifest.json")
    if not parent.is_file():
        parent = source.parent / "manifest.json"
    parent_path = parent if parent.is_file() else None
    if args.command == "run":
        scorers, info, cache = scoring(args)
        try:
            result = run(
                source,
                out_dir=args.output,
                eval_sets=args.evaluation,
                keep=args.keep,
                seed=args.seed,
                scorers=scorers,
                selector=selecting(args),
                parameters={
                    **info,
                    "selection": {
                        **selection_parameters(args),
                        "by": args.by,
                        "strata": args.strata,
                        "threshold": args.threshold,
                        "embedder": args.embedder,
                    },
                },
            )
        finally:
            if cache:
                cache.close()
        print(
            f"selected {len(result.pipeline.selection.examples)} / "
            f"{len(result.pipeline.scored.examples)}"
        )
        return 0 if result.pipeline.contamination.ok else 1
    output = args.output
    if args.command == "select" and output.suffix != ".jsonl":
        output = output / "selected.jsonl"
    manifest_path = output.parent / (output.stem + ".manifest.json")
    protected = {source.resolve(), parent.resolve()}
    decisions_path = output.parent / (output.stem + ".decisions.json")
    if any(p.resolve() in protected for p in (output, manifest_path, decisions_path)):
        raise ValueError("output would overwrite input or parent")
    if args.command == "score":
        scorers, info, cache = scoring(args)
        try:
            result_score = score_dataset(source, scorers=scorers)
        finally:
            if cache:
                cache.close()
        write_scored(output, result_score.examples)
        steps = ({"name": "score", **info},)
        count = len(result_score.examples)
        outputs: tuple[tuple[Path, int | None], ...] = ((output, count),)
    else:
        selected = select_subset(source, keep=args.keep, selector=selecting(args))
        write_scored(output, selected.examples, selected.decisions)
        decisions_path.write_text(
            json.dumps([asdict(d) for d in selected.decisions], sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        count = len(selected.examples)
        steps = ({"name": "select", **selection_parameters(args), "out": count},)
        outputs = ((output, count), (decisions_path, len(selected.decisions)))
    record_manifest((source,), (), outputs, steps, manifest_path, parent_path)
    print(f"wrote {count} records to {output}")
    return 0
