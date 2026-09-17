"""Render trained-model quality against retention budget from measured judgments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("experiments/results/retention-curve")


def main() -> None:
    primary = json.loads((ROOT / "primary.summary.json").read_text())
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for arm, color in (("selected", "#176B87"), ("random", "#B85C38")):
        values, lower, upper = [], [], []
        for budget in (100, 200, 400, 600, 800, 1000):
            if budget == 1000:
                value, lo, hi = 0.5, 0.5, 0.5
            else:
                result = json.loads((ROOT / f"{arm}-{budget}-42.summary.json").read_text())
                value = result["preference_score_ties_half"]
                lo, hi = result["prompt_cluster_bootstrap_95"]
            rows.append(
                {"arm": arm, "retention": budget / 1000, "preference": value, "ci95": [lo, hi]}
            )
            values.append(value)
            lower.append(value - lo)
            upper.append(hi - value)
        ax.errorbar(
            [10, 20, 40, 60, 80, 100],
            values,
            yerr=[lower, upper],
            label=arm,
            color=color,
            marker="o",
            capsize=4,
        )
    ax.axhline(0.5, color="#666", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Training records retained (%)",
        ylabel="Preference versus full dataset (ties = ½)",
        title="Alpaca-1k · measured downstream quality curve\n"
        "Exploratory: seed 42, 100 fresh evaluation prompts",
    )
    ax.set_ylim(0, 1)
    ax.set_xticks([10, 20, 40, 60, 80, 100])
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "curve.png", dpi=180)
    fig.savefig(ROOT / "curve.svg")
    p = ROOT / "curve.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    (ROOT / "curve.json").write_text(json.dumps(rows, indent=2) + "\n")
    counts = primary["counts"]
    lo, hi = primary["prompt_cluster_bootstrap_95"]
    lines = [
        "# Corrected retention-budget pilot",
        "",
        "Completed using the fixed [plan](plan.json), declared before judging.",
        "",
        "## Primary comparison: selected 20% versus random 20%",
        "",
        f"Three seeds, 100 fresh prompts: **{counts['selected']} wins, "
        f"{counts['baseline']} losses, {counts['tie']} ties**.",
        f"Preference (ties count as half): **{primary['preference_score_ties_half']:.1%}**, "
        f"clustered 95% interval **{lo:.1%}–{hi:.1%}**.",
        (
            "The local superiority criterion was met."
            if primary["criterion_met"]
            else "The superiority criterion was **not met**."
        ),
        "",
        "## Quality versus retention budget",
        "",
        "![Measured downstream curve](curve.png)",
        "",
        "Each point compares generated answers from a trained subset model against",
        "the model trained on all 1,000 records. At 100%, both are the same model",
        "and the score is 50% by definition. The curve uses one training seed; its",
        "intervals resample prompts and do not measure variation across training seeds.",
        "",
        "| Retained | Selected preference vs full | Random preference vs full |",
        "| --- | ---: | ---: |",
    ]
    for fraction in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        points = [r for r in rows if r["retention"] == fraction]
        lines.append(
            f"| {fraction:.0%} | {points[0]['preference']:.1%} | {points[1]['preference']:.1%} |"
        )
    lines += [
        "",
        "## Corrections and limits",
        "",
        "Training now learns the end-of-response token. The two judge orders",
        "use the paper's aggregation policy: a win plus a tie counts as a win;",
        "opposing wins remain a tie. Old and new evaluation sets are disjoint.",
        "The previous pilot is retained unchanged. Selection settings were not",
        "tuned against either evaluation set. Individual decisions and metadata",
        "are included; raw dataset text and generated answers remain local.",
        "",
        "This measures downstream quality at six budgets, but adapts the paper",
        "to a 135M instruction model, 1,000 records and one epoch. It does not",
        "reproduce the original LLaMA-7B pre-experience stage, training or judge.",
        "The single quantized judge is not human ground truth. A null result",
        "does not establish equivalence or justify changing the acceptance threshold.",
        "",
        "Reproduce: `python experiments/retention_curve.py` then",
        "`python experiments/report_retention_curve.py`.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
