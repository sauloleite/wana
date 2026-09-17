"""Prepare equal-size selected/random arms and a disjoint evaluation set."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from wana import check_contamination, select_subset
from wana.adapters.embedding.hashing import HashingEmbedder
from wana.adapters.io.jsonl import parse_record
from wana.adapters.io.scored import write_scored
from wana.adapters.report.json import serialize
from wana.adapters.selection.diverse_greedy import DiverseGreedySelector
from wana.domain.score import Score, ScoredExample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = []
    indices = []
    for line in args.scores.read_text().splitlines():
        item = json.loads(line)
        indices.append(item["index"])
        rows.append(
            ScoredExample(
                parse_record(data[item["index"]], str(item["index"])),
                tuple(Score(s["name"], s["value"], tuple(s["flags"])) for s in item["scores"]),
            )
        )
    selection = select_subset(rows, keep=0.2, selector=DiverseGreedySelector(HashingEmbedder()))
    random_rows = tuple(
        rows[i]
        for i in sorted(random.Random(args.seed).sample(range(len(rows)), len(selection.examples)))
    )
    remaining = sorted(set(range(len(data))) - set(indices))
    evaluation = [
        parse_record(data[i], str(i))
        for i in sorted(random.Random(args.seed).sample(remaining, 100))
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    write_scored(args.output / "full.jsonl", rows)
    write_scored(args.output / "selected.jsonl", selection.examples, selection.decisions)
    write_scored(args.output / "random.jsonl", random_rows)
    with (args.output / "evaluation.jsonl").open("w", encoding="utf-8") as stream:
        for e in evaluation:
            stream.write(json.dumps(e.original, ensure_ascii=False) + "\n")
    report = check_contamination([r.example for r in rows], eval_sets=[evaluation])
    (args.output / "contamination.json").write_text(serialize(report))
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "full": len(rows),
                "selected": len(selection.examples),
                "random": len(random_rows),
                "evaluation": len(evaluation),
                "selected_ids": [r.example.id for r in selection.examples],
                "random_ids": [r.example.id for r in random_rows],
                "contamination_ok": report.ok,
                "decisions": [asdict(d) for d in selection.decisions],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Prepared {len(selection.examples)} selected and random examples; "
        f"contamination ok={report.ok}"
    )


if __name__ == "__main__":
    main()
