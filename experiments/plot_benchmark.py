"""Plot the measured pilot comparisons, including ties and clustered uncertainty."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path("experiments/results/benchmark")
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    for y, baseline in enumerate(("random", "full")):
        summary = json.loads((root / f"selected-vs-{baseline}.summary.json").read_text())
        value = summary["preference_score_ties_half"]
        low, high = summary["prompt_cluster_bootstrap_95"]
        ax.errorbar(
            value,
            y,
            xerr=[[value - low], [high - value]],
            fmt="o",
            capsize=6,
            color="#176B87",
            markersize=9,
        )
        ax.annotate(
            f"{value:.1%} [{low:.1%}, {high:.1%}]",
            (value, y),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
        )
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_yticks([0, 1], ["Selected 20% vs random 20%", "Selected 20% vs full 100%"])
    ax.set_xlim(0.25, 0.75)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Preference for selected subset (ties count as ½)")
    ax.set_title(
        "Alpaca-1k · SmolLM2-135M pilot\n100 prompts · 3 seeds · prompt-cluster bootstrap 95%"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "comparisons.png", dpi=180)
    fig.savefig(root / "comparisons.svg")


if __name__ == "__main__":
    main()
