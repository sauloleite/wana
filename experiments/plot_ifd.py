"""Plot measured IFD distribution; requires matplotlib (experiment-only)."""

import argparse
import json
from pathlib import Path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (args.directory / "scores.jsonl").read_text().splitlines()]
    values = [next(s["value"] for s in row["scores"] if s["name"] == "ifd") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].hist(values, bins=35, color="#167d9a")
    axes[0].axvline(1, color="#bd452d", linestyle="--", label="IFD = 1")
    axes[0].set(xlabel="IFD", ylabel="Examples", title="Alpaca-1k: measured scores")
    axes[0].legend()
    curve = summary["selection_curve"]
    axes[1].plot(
        [x["fraction"] * 100 for x in curve],
        [x["mean_ifd"] for x in curve],
        marker="o",
        color="#167d9a",
    )
    axes[1].set(
        xlabel="Requested retention (%)",
        ylabel="Mean selected IFD",
        title="Ranking behavior, not training quality",
    )
    fig.suptitle("SmolLM2-135M Q4_1 · CPU · seed 42 · 1,000 examples")
    fig.savefig(args.directory / "ifd.png", dpi=180)
    fig.savefig(args.directory / "ifd.svg")


if __name__ == "__main__":
    main()
