"""Render the pilot's measured outcomes without turning null results into success."""

import json
from pathlib import Path

from wana.adapters.io.provenance import digest


def main() -> None:
    root = Path("experiments/results/benchmark")
    plan = json.loads((root / "plan.json").read_text())
    lines = [
        "# Alpaca-1k trained-model pilot",
        "",
        "Three full-parameter fine-tuning seeds (42, 43, 44), one epoch, learning rate",
        "2e-5, SmolLM2-135M-Instruct, Metal. The selected arm uses 200 examples ranked",
        "by IFD with hashing TF-IDF diversity; the random arm has 200 examples and the",
        "full arm 1,000. Random subsets vary by seed; the selected subset is fixed.",
        "",
        "All arms generate greedy answers (128 new tokens maximum) for the same 100",
        "held-out Alpaca prompts. Qwen3-4B-Instruct-2507 Q4_K_M judges each pair in both",
        "orders with blinded arm labels. Order disagreement becomes a tie. The",
        "reference answer assists the judge; it is not assumed to be infallible.",
        "",
        "## Results",
        "",
        "| Comparison | Selected wins | Baseline wins | Ties | "
        "Preference (ties = ½) | Clustered 95% interval |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    results = {}
    for baseline in ("random", "full"):
        result = json.loads((root / f"selected-vs-{baseline}.summary.json").read_text())
        results[baseline] = result
        c = result["counts"]
        low, high = result["prompt_cluster_bootstrap_95"]
        lines.append(
            f"| Selected vs {baseline} | {c['selected']} | {c['baseline']} | {c['tie']} | "
            f"{result['preference_score_ties_half']:.1%} | {low:.1%}–{high:.1%} |"
        )
    met = results["random"]["criterion_met"]
    identical = 0
    for seed in plan["training_seeds"]:
        pairs = [
            [
                json.loads(line)
                for line in (root / f"{arm}-{seed}.answers.jsonl").read_text().splitlines()
            ]
            for arm in ("selected", "random")
        ]
        identical += sum(a["answer"] == b["answer"] for a, b in zip(*pairs, strict=True))
    lines.extend(
        [
            "",
            f"The two arms produced identical answers in {identical}/300 primary pairs, "
            "which are counted as ties without calling the judge.",
            "",
            f"Changing answer order changed the judge's decision in "
            f"{results['random']['position_disagreements']}/300 primary pairs and "
            f"{results['full']['position_disagreements']}/300 secondary pairs. "
            "These pairs are counted as ties.",
            "",
            (
                "The primary comparison met the predeclared local superiority criterion."
                if met
                else "The primary comparison did **not** meet the superiority criterion."
            ),
            "The criterion requires the lower endpoint of the prompt-cluster bootstrap",
            "interval to exceed 50%. This is an exploratory small-model pilot with one",
            "quantized model judge, not an exact replication of the IFD paper or a human",
            "quality evaluation. Failure to establish superiority does not establish",
            "equivalence. Hyperparameters and selection were not tuned to judge outcomes.",
            "",
            "![Measured comparisons](comparisons.png)",
            "",
            "## Reproduction and audit",
            "",
            "The predeclared settings are in [plan.json](plan.json). The scripts and",
            "commands are described in [experiments/README.md](../../README.md). Raw",
            "datasets, answers and checkpoints remain local; hashes, training metadata,",
            "individual decisions and summaries are versioned. No upstream dataset text",
            "is redistributed. EXACT/NEAR checks found no split overlap; semantic or",
            "pretraining contamination is not ruled out.",
            "",
            "The bootstrap resamples the 100 prompts and preserves their three seed",
            "results together (10,000 replicates, seed 42). It quantifies uncertainty over",
            "these prompts, conditional on these three training runs and this judge.",
            "",
            f"Judge SHA-256: `{plan['judge_sha256']}`.",
            "",
        ]
    )
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    paths = sorted(Path("experiments/checkpoints").glob("*/*.safetensors"))
    paths += sorted(root.glob("*.answers.jsonl"))
    paths += sorted(Path("experiments").glob("*.py"))
    (root / "artifacts.json").write_text(
        json.dumps({str(path): digest(path).sha256 for path in paths}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
