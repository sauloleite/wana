"""Judge all completed benchmark arms and aggregate each comparison separately."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    root = Path("experiments/results/benchmark")
    for baseline in ("random", "full"):
        judgments = []
        for seed in args.seeds:
            paths = [root / f"{arm}-{seed}.answers.jsonl" for arm in ("selected", baseline)]
            if not all(p.with_suffix(".metadata.json").exists() for p in paths):
                raise ValueError("finish run_benchmark.py before judging")
            output = root / f"selected-vs-{baseline}-{seed}.judgments.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "experiments/judge_pairs.py",
                    "--selected",
                    str(paths[0]),
                    "--random",
                    str(paths[1]),
                    "--baseline-name",
                    baseline,
                    "--seed",
                    str(seed),
                    "--model",
                    str(args.model),
                    "--output",
                    str(output),
                ],
                check=True,
            )
            judgments.append(str(output))
        subprocess.run(
            [
                sys.executable,
                "experiments/aggregate_benchmark.py",
                *judgments,
                "-o",
                str(root / f"selected-vs-{baseline}.summary.json"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
