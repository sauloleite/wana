"""Prompt-cluster bootstrap, preserving seed correlation and ties."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def aggregate(rows: list[dict[str, object]], *, samples: int = 10000) -> dict[str, object]:
    if not rows:
        raise ValueError("no judgments")
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples required")
    baselines = {row.get("baseline") for row in rows}
    judges = {row.get("judge_sha256") for row in rows}
    if len(baselines) != 1 or not baselines <= {"random", "full"} or len(judges) != 1:
        raise ValueError("aggregate one baseline and one judge at a time")
    grouped: dict[str, list[float]] = defaultdict(list)
    seen = set()
    counts = {"selected": 0, "baseline": 0, "tie": 0}
    disagreements = 0
    seeds = set()
    for row in rows:
        key = (str(row["id"]), int(row["seed"]))
        if key in seen:
            raise ValueError("duplicate prompt/seed")
        seen.add(key)
        seeds.add(key[1])
        winner = row["winner"]
        if winner not in ("selected", row["baseline"], "tie"):
            raise ValueError("invalid winner")
        grouped[key[0]].append(1.0 if winner == "selected" else 0.5 if winner == "tie" else 0.0)
        counts[
            "selected" if winner == "selected" else "tie" if winner == "tie" else "baseline"
        ] += 1
        disagreements += bool(row.get("order_disagreement"))
    sizes = {len(values) for values in grouped.values()}
    if sizes != {len(seeds)}:
        raise ValueError("all prompts must have all seeds")
    means = [sum(values) / len(values) for _, values in sorted(grouped.items())]
    rng = random.Random(42)
    bootstrap = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(samples))
    lower, upper = (
        bootstrap[int(0.025 * samples)],
        bootstrap[min(samples - 1, int(0.975 * samples))],
    )
    return {
        "baseline": next(iter(baselines)),
        "prompts": len(grouped),
        "seeds": sorted(seeds),
        "judgments": len(rows),
        "counts": counts,
        "position_disagreements": disagreements,
        "preference_score_ties_half": sum(means) / len(means),
        "prompt_cluster_bootstrap_95": [lower, upper],
        "bootstrap_samples": samples,
        "criterion_met": lower > 0.5,
        "scope": "Local small-model pilot; single quantized model judge, not human ground truth",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for p in args.files for line in p.read_text().splitlines()]
    result = aggregate(rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
