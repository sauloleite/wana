"""Execute the predeclared three-arm benchmark sequentially, resuming completed arms."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    args = parser.parse_args()
    full = Path("experiments/results/subsets/full.jsonl")
    lines = full.read_text(encoding="utf-8").splitlines()
    evaluation = Path("experiments/results/subsets/evaluation.jsonl")
    result = Path("experiments/results/benchmark")
    result.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        random_path = Path(f"experiments/results/subsets/random-{seed}.jsonl")
        indices = sorted(random.Random(seed).sample(range(len(lines)), 200))
        random_path.write_text("\n".join(lines[i] for i in indices) + "\n", encoding="utf-8")
        for arm in ("selected", "random", "full"):
            dataset = (
                Path(f"experiments/results/subsets/{arm}.jsonl") if arm != "random" else random_path
            )
            checkpoint = Path(f"experiments/checkpoints/{arm}-{seed}")
            metadata = checkpoint / "training.json"
            if metadata.exists():
                completed = json.loads(metadata.read_text())
                if completed["dataset_sha256"] != hashlib.sha256(dataset.read_bytes()).hexdigest():
                    raise ValueError("dataset changed since training; use a fresh checkpoint")
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/train_subset.py",
                        "--legacy-no-eos",
                        str(dataset),
                        "--model",
                        "experiments/models/smollm2",
                        "--output",
                        str(checkpoint),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                    ],
                    check=True,
                )
            (result / f"{arm}-{seed}.training.json").write_text(metadata.read_text())
            predictions = result / f"{arm}-{seed}.answers.jsonl"
            if not predictions.with_suffix(".metadata.json").exists():
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/generate_answers.py",
                        "--checkpoint",
                        str(checkpoint),
                        "--evaluation",
                        str(evaluation),
                        "--output",
                        str(predictions),
                        "--device",
                        args.device,
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
