"""Compare ranking consistency between installed GGUF and local Transformers weights."""

import json
import random
import statistics
from pathlib import Path

import torch

from wana.adapters.io.jsonl import parse_record
from wana.adapters.logprob.bundled import BundledLogProbProvider
from wana.adapters.logprob.transformers_cpu import TransformersCPU
from wana.adapters.scoring.ifd import IFDScorer


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        statistics.mean(i for i, other in enumerate(ordered) if other == value) for value in values
    ]


def main() -> None:
    torch.set_num_threads(4)
    records = json.loads(Path("experiments/data/alpaca_data.json").read_text())
    summary = json.loads(Path("experiments/results/alpaca-1k/summary.json").read_text())
    indices = sorted(random.Random(17).sample(summary["sample_indices"], 20))
    gguf = IFDScorer(BundledLogProbProvider())
    transformers = IFDScorer(TransformersCPU("experiments/models/smollm2", max_tokens=2048))
    rows = []
    for index in indices:
        example = parse_record(records[index], str(index))
        rows.append(
            {
                "id": index,
                "gguf_ifd": gguf.score(example)[0].value,
                "transformers_ifd": transformers.score(example)[0].value,
            }
        )
    result = {
        "sample_seed": 17,
        "records": rows,
        "spearman": statistics.correlation(
            ranks([r["gguf_ifd"] for r in rows]), ranks([r["transformers_ifd"] for r in rows])
        ),
        "scope": "Adapter/quantization consistency audit on training records; "
        "not downstream quality.",
    }
    Path("experiments/results/scorer-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["spearman"])


if __name__ == "__main__":
    main()
