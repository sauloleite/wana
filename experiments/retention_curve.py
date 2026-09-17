"""A fresh, fixed-budget replication pilot; never tune on the held-out results."""

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

from wana import check_contamination, select_subset
from wana.adapters.embedding.hashing import HashingEmbedder
from wana.adapters.io.jsonl import JsonlReader, parse_record
from wana.adapters.io.scored import from_example, write_scored
from wana.adapters.report.json import serialize
from wana.adapters.selection.diverse_greedy import DiverseGreedySelector

ROOT = Path("experiments/results/retention-curve")
DATA = Path("experiments/data/retention-curve")
MODEL = Path("experiments/models/smollm2")
JUDGE = Path("experiments/models/judge/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
BUDGETS = (100, 200, 400, 600, 800, 1000)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(script: str, *args: object) -> None:
    subprocess.run([sys.executable, f"experiments/{script}.py", *map(str, args)], check=True)


def prepare() -> dict[str, object]:
    ROOT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    original = Path("experiments/data/alpaca_data.json")
    records = json.loads(original.read_text())
    summary = json.loads(Path("experiments/results/alpaca-1k/summary.json").read_text())
    if sha(original) != summary["dataset_sha256"]:
        raise ValueError("dataset differs from the scored dataset")
    training_ids = set(summary["sample_indices"])
    remaining = sorted(set(range(len(records))) - training_ids)
    old_eval = set(random.Random(42).sample(remaining, 100))
    evaluation_ids = sorted(random.Random(20260917).sample(sorted(set(remaining) - old_eval), 100))
    evaluation = [parse_record(records[i], str(i)) for i in evaluation_ids]
    rows = tuple(
        from_example(e) for e in JsonlReader().read("experiments/results/subsets/full.jsonl")
    )
    report = check_contamination([r.example for r in rows], eval_sets=[evaluation])
    if not report.ok:
        raise ValueError("new evaluation overlaps training; investigate before running")
    (ROOT / "contamination.json").write_text(serialize(report))
    evaluation_path = DATA / "evaluation.jsonl"
    evaluation_path.write_text("".join(json.dumps(e.original) + "\n" for e in evaluation))
    for seed in (42, 43, 44):
        permutation = random.Random(seed).sample(range(len(rows)), len(rows))
        for budget in BUDGETS if seed == 42 else (200,):
            for arm in ("full",) if budget == 1000 else ("selected", "random"):
                if arm == "full":
                    chosen = rows
                elif arm == "random":
                    ids = set(permutation[:budget])
                    chosen = tuple(r for i, r in enumerate(rows) if i in ids)
                else:
                    result = select_subset(
                        rows,
                        keep=budget,
                        selector=DiverseGreedySelector(HashingEmbedder()),
                    )
                    chosen = result.examples
                    if len(chosen) != budget:
                        raise ValueError("diversity reduced the requested budget")
                write_scored(DATA / f"{arm}-{budget}-{seed}.jsonl", chosen)
    plan = {
        "study": "corrected local replication pilot; original results are preserved",
        "budgets": BUDGETS,
        "curve_seed": 42,
        "primary_seeds": [42, 43, 44],
        "primary_comparison": "selected 200 versus random 200",
        "curve_comparison": "each arm versus full 1000, same seed",
        "model_snapshot": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "model_sha256": sha(MODEL / "model.safetensors"),
        "judge_sha256": sha(JUDGE),
        "evaluation_ids": evaluation_ids,
        "excluded_previous_evaluation_ids": sorted(old_eval),
        "data_sha256": sha(original),
        "evaluation_sha256": sha(evaluation_path),
        "epochs": 1,
        "learning_rate": 2e-5,
        "device": "mps",
        "target_eos": True,
        "max_new_tokens": 128,
        "order_policy": "paper",
        "criterion": "primary prompt-cluster bootstrap lower bound > 0.5",
        "limitations": "135M instruct, 1k records, one epoch; not exact LLaMA-7B replication",
        "dataset_hashes": {p.name: sha(p) for p in sorted(DATA.glob("*.jsonl"))},
    }
    plan_path = ROOT / "plan.json"
    serialized = json.dumps(plan, indent=2) + "\n"
    if plan_path.exists() and plan_path.read_text() != serialized:
        raise ValueError("predeclared protocol changed; use a new study directory")
    plan_path.write_text(serialized)
    return plan


def main() -> None:
    plan = prepare()
    # Complete the primary comparison first, then the secondary budget curve.
    arms = [(arm, 200, seed) for seed in (42, 43, 44) for arm in ("selected", "random")]
    arms += [("full", 1000, 42)]
    arms += [
        (arm, n, 42) for n in BUDGETS if n not in (200, 1000) for arm in ("selected", "random")
    ]
    for arm, budget, seed in arms:
        name = f"{arm}-{budget}-{seed}"
        dataset = DATA / f"{name}.jsonl"
        checkpoint = Path("experiments/checkpoints/retention-curve") / name
        training = checkpoint / "training.json"
        if training.exists():
            saved = json.loads(training.read_text())
            expected = {
                "dataset_sha256": sha(dataset),
                "seed": seed,
                "epochs": plan["epochs"],
                "target_eos": True,
                "learning_rate": plan["learning_rate"],
                "model": str(MODEL),
            }
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError(f"training identity changed: {name}")
        else:
            call(
                "train_subset",
                dataset,
                "--model",
                MODEL,
                "--output",
                checkpoint,
                "--seed",
                seed,
                "--epochs",
                plan["epochs"],
                "--device",
                plan["device"],
            )
        (ROOT / f"{name}.training.json").write_text(training.read_text())
        predictions = ROOT / f"{name}.answers.jsonl"
        metadata = predictions.with_suffix(".metadata.json")
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["dataset_sha256"] != plan["evaluation_sha256"] or saved[
                "predictions_sha256"
            ] != sha(predictions):
                raise ValueError(f"generation identity changed: {name}")
        else:
            call(
                "generate_answers",
                "--checkpoint",
                checkpoint,
                "--evaluation",
                DATA / "evaluation.jsonl",
                "--output",
                predictions,
                "--device",
                plan["device"],
            )
        if arm == "random" and budget == 200:
            call(
                "judge_pairs",
                "--selected",
                ROOT / f"selected-200-{seed}.answers.jsonl",
                "--random",
                predictions,
                "--model",
                JUDGE,
                "--seed",
                seed,
                "--order-policy",
                "paper",
                "--output",
                ROOT / f"primary-{seed}.judgments.jsonl",
            )
    call(
        "aggregate_benchmark",
        *(ROOT / f"primary-{s}.judgments.jsonl" for s in (42, 43, 44)),
        "-o",
        ROOT / "primary.summary.json",
    )
    for arm, budget, seed in arms:
        if seed != 42 or arm == "full":
            continue
        name = f"{arm}-{budget}-{seed}"
        call(
            "judge_pairs",
            "--selected",
            ROOT / f"{name}.answers.jsonl",
            "--random",
            ROOT / "full-1000-42.answers.jsonl",
            "--baseline-name",
            "full",
            "--model",
            JUDGE,
            "--seed",
            seed,
            "--order-policy",
            "paper",
            "--output",
            ROOT / f"{name}.judgments.jsonl",
        )
        call(
            "aggregate_benchmark",
            ROOT / f"{name}.judgments.jsonl",
            "-o",
            ROOT / f"{name}.summary.json",
        )


if __name__ == "__main__":
    main()
