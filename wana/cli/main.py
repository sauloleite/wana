"""CLI composition root and filesystem provenance collection."""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from wana import __version__
from wana.adapters.io.jsonl import JsonlReader
from wana.adapters.matching.minhash import MinHashMatcher
from wana.adapters.matching.ngram import NgramMatcher
from wana.adapters.report.json import serialize
from wana.adapters.report.terminal import render
from wana.application.build_manifest import build_manifest
from wana.application.check_contamination import check_replayed
from wana.domain.contamination import Level
from wana.domain.manifest import Digest, Step


def digest(path: Path, records: int | None = None) -> Digest:
    """Hash raw file bytes, including compression when present."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return Digest(str(path), value.hexdigest(), records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wana", description="Audit SFT train/evaluation overlap.")
    parser.add_argument("--version", action="version", version=f"wana {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check JSONL train/evaluation contamination")
    check.add_argument("train", nargs="+", type=Path)
    check.add_argument("--eval", dest="evaluation", action="append", required=True, type=Path)
    check.add_argument("-o", "--out-dir", type=Path, default=Path("wana-check"))
    check.add_argument("--markdown", action="store_true")
    check.add_argument("--semantic", action="store_true")
    check.add_argument("--embedding-model")
    check.add_argument("--tokenizer")
    check.add_argument("--semantic-threshold", type=float, default=0.9)
    check.add_argument("--ngram", type=int, default=13)
    check.add_argument("--threshold", type=float, default=0.8)
    check.add_argument("--shingle", type=int, default=5)
    check.add_argument("--ignore-template", action="append", default=[], metavar="LITERAL_TEXT")
    check.add_argument(
        "--fail-on", nargs="*", choices=["EXACT", "NEAR", "SEMANTIC"], default=["EXACT", "NEAR"]
    )
    check.add_argument(
        "--parent", type=Path, help="Parent manifest (auto-detected beside first input)"
    )
    from wana.cli.curation import add_commands

    add_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 for pass, 1 for contamination and 2 for invalid input or I/O."""
    args = _parser().parse_args(argv)
    try:
        if args.command != "check":
            from wana.cli.curation import execute

            return execute(args)
        if "SEMANTIC" in args.fail_on and not args.semantic:
            raise ValueError("--fail-on SEMANTIC requires --semantic")
        ignored = tuple(args.ignore_template)
        from wana.ports.matcher import Matcher

        matchers: tuple[Matcher, ...] = (
            NgramMatcher(args.ngram, ignore_template=ignored),
            MinHashMatcher(args.threshold, shingle=args.shingle, ignore_template=ignored),
        )
        if args.semantic:
            from wana.adapters.embedding.onnx import OnnxEmbedder
            from wana.adapters.matching.embedding import EmbeddingMatcher

            if not args.embedding_model or not args.tokenizer:
                raise ValueError("--semantic requires --embedding-model and --tokenizer")
            matchers = (
                *matchers,
                EmbeddingMatcher(
                    OnnxEmbedder(args.embedding_model, args.tokenizer), args.semantic_threshold
                ),
            )
        reader = JsonlReader()
        testing = [tuple(reader.read(str(path))) for path in args.evaluation]
        inputs = tuple(digest(path, sum(1 for _ in reader.read(str(path)))) for path in args.train)
        eval_sets = tuple(
            digest(path, len(rows)) for path, rows in zip(args.evaluation, testing, strict=True)
        )
        parent_path = args.parent
        if parent_path is None:
            candidate = args.train[0].parent / "manifest.json"
            parent_path = candidate if candidate.is_file() else None
        parent = None
        if parent_path is not None:
            if not isinstance(json.loads(parent_path.read_text(encoding="utf-8")), dict):
                raise ValueError("parent manifest must be a JSON object")
            parent = digest(parent_path)
        report_path = args.out_dir / "contamination.json"
        manifest_path = args.out_dir / "manifest.json"
        protected = {p.resolve() for p in [*args.train, *args.evaluation]}
        if parent_path is not None:
            protected.add(parent_path.resolve())
        if any(p.resolve() in protected for p in (report_path, manifest_path)):
            raise ValueError(
                "output would overwrite an input or parent manifest; choose another -o"
            )
        report = check_replayed(
            lambda: (e for path in args.train for e in reader.read(str(path))),
            (e for rows in testing for e in rows),
            matchers=matchers,
            fail_on=tuple(Level(level) for level in args.fail_on),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialize(report), encoding="utf-8", newline="\n")
        step = Step(
            "check",
            tuple(m.name for m in matchers),
            args.ngram,
            args.threshold,
            args.shingle,
            64,
            16,
            ignored,
            tuple(args.fail_on),
            len(report.hits),
        )
        manifest = build_manifest(
            version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            inputs=inputs,
            eval_sets=eval_sets,
            steps=(
                {
                    **asdict(step),
                    "semantic_threshold": args.semantic_threshold,
                    "embedding_model": asdict(digest(Path(args.embedding_model).resolve())),
                    "tokenizer": asdict(digest(Path(args.tokenizer).resolve())),
                },
            )
            if args.semantic
            else (step,),
            outputs=(digest(report_path, len(report.hits)),),
            parent=parent,
        )
        manifest_path.write_text(serialize(manifest), encoding="utf-8", newline="\n")
        if args.markdown:
            from wana.adapters.report.markdown import render_markdown

            print(render_markdown(report))
        else:
            print(render(report))
        return 0 if report.ok else 1
    except (OSError, ValueError, EOFError) as exc:
        print(f"wana: {exc}", file=sys.stderr)
        return 2
