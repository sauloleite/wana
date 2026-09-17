"""Reproducible Alpaca-1k IFD distribution (not a downstream training benchmark)."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from wana.adapters.io.jsonl import parse_record
from wana.adapters.logprob.bundled import BundledLogProbProvider
from wana.adapters.logprob.cache import CachedLogProbProvider
from wana.adapters.scoring.ifd import IFDScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    indices = sorted(random.Random(args.seed).sample(range(len(records)), min(1000, len(records))))
    args.output.mkdir(parents=True, exist_ok=True)
    model = BundledLogProbProvider(max_tokens=2048)
    identity = json.dumps(
        {
            "model_sha256": model.revision,
            "template": model.chat_template,
            "max_tokens": 2048,
            "adapter": "bundled-v1",
        },
        sort_keys=True,
    )
    provider = CachedLogProbProvider(model, args.output / "cache.sqlite3", identity)
    scorer = IFDScorer(provider)
    values, errors = [], []
    start = time.monotonic()
    with (args.output / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for n, index in enumerate(indices):
            try:
                scores = scorer.score(parse_record(records[index], str(index)))
                value = scores[0].value
                values.append(value)
                stream.write(
                    json.dumps({"index": index, "scores": [asdict(s) for s in scores]}) + "\n"
                )
                stream.flush()
            except ValueError as exc:
                errors.append({"index": index, "error": str(exc)})
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/1000, elapsed={time.monotonic() - start:.1f}s", flush=True)
    provider.close()
    ordered = sorted(values)
    eligible = sorted((x for x in values if x <= 1), reverse=True)
    summary = {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": model.model_name,
        "model_sha256": model.revision,
        "seed": args.seed,
        "sample_indices": indices,
        "scored": len(values),
        "errors": errors,
        "ifd_above_one": sum(v > 1 for v in values),
        "quantiles": {
            str(p): ordered[int(p * (len(ordered) - 1))] for p in (0, 0.25, 0.5, 0.75, 1)
        },
        "selection_curve": [
            {
                "fraction": p,
                "count": len(eligible[: int(len(values) * p)]),
                "mean_ifd": sum(eligible[: int(len(values) * p)])
                / max(1, len(eligible[: int(len(values) * p)])),
            }
            for p in (0.1, 0.2, 0.4, 0.6, 0.8, 1)
        ],
        "elapsed_seconds": time.monotonic() - start,
        "claim": "IFD scoring distribution only; no reproduction of downstream quality curve",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("sample_indices", "errors")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
