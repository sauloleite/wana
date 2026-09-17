"""Download pinned public snapshots for the optional integration tests."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODELS = (
    (
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ),
        "smollm2",
    ),
    (
        "Xenova/all-MiniLM-L6-v2",
        "751bff37182d3f1213fa05d7196b954e230abad9",
        ("onnx/model_quantized.onnx", "tokenizer.json"),
        "minilm",
    ),
)


def main() -> None:
    files = []
    for repo, revision, names, directory in MODELS:
        for name in names:
            path = Path("experiments/models") / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            if not path.exists():
                with urlopen(url, timeout=120) as response:
                    path.write_bytes(response.read())
            files.append({"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    Path("experiments/models/sources.json").write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    main()
