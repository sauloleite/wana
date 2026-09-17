"""Aggregate paired, blinded judgments of trained-model outputs.

Input JSONL: {id, winner: selected|random|tie, judge_model, seed}.
A win rate confidence interval above 0.5 is required; no result is manufactured.
"""

import argparse
import json
import math
from pathlib import Path


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    seen: set[tuple[str, str]] = set()
    wins = losses = ties = 0
    for row in records:
        key = (str(row["id"]), str(row["seed"]))
        if key in seen:
            raise ValueError("duplicate evaluation id and seed")
        seen.add(key)
        if not row.get("judge_model"):
            raise ValueError("judge identity required")
        winner = row["winner"]
        if winner == "selected":
            wins += 1
        elif winner == "random":
            losses += 1
        elif winner == "tie":
            ties += 1
        else:
            raise ValueError("winner must be selected, random or tie")
    n = wins + losses
    if not n:
        return {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "non_tie_win_rate": None,
            "wilson_95": [0.0, 1.0],
            "criterion_met": False,
            "caveat": "No non-tied judgments; inconclusive.",
        }
    p = wins / n
    z = 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_tie_win_rate": p,
        "wilson_95": [center - radius, center + radius],
        "criterion_met": center - radius > 0.5,
        "caveat": (
            "Descriptive interval; repeated prompts across seeds are correlated. "
            "Use a clustered analysis for publication."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judgments", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.judgments.read_text().splitlines() if line.strip()]
    args.output.write_text(json.dumps(summarize(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
