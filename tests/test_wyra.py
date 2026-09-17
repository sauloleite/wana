"""Real Wyra CLI integration, run in the dedicated CI integration job."""

import importlib.util
import json
import subprocess
import sys

import pytest

from wana import run, verify_manifest
from wana.adapters.logprob.fake import FakeLogProbProvider
from wana.adapters.scoring.ifd import IFDScorer


def test_wyra_pipeline_with_planted_leakage(tmp_path):
    if importlib.util.find_spec("wyra") is None:
        pytest.skip("install wyra to run the real integration")
    source = tmp_path / "source.md"
    source.write_text(
        "\n\n".join(
            f"# Topic {i}\n\nThis document explains topic {i} in sufficient detail to create "
            "a training example about deterministic dataset quality and reliable provenance "
            "for language models."
            for i in range(20)
        )
    )
    dataset = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from wyra.cli import main; raise SystemExit(main())",
            "build",
            str(source),
            "-o",
            str(dataset),
            "--valid",
            ".3",
        ],
        check=True,
    )
    evaluation = dataset / "validation.jsonl"
    with (dataset / "train.jsonl").open("a") as stream:
        stream.write(evaluation.read_text().splitlines()[0] + "\n")
    result = run(
        dataset,
        out_dir=tmp_path / "out",
        eval_sets=[evaluation],
        keep=1.0,
        scorers=[IFDScorer(FakeLogProbProvider())],
    )
    assert not result.pipeline.contamination.ok
    assert result.manifest.parent.path == str((dataset / "manifest.json").resolve())
    assert verify_manifest(tmp_path / "out/manifest.json").ok
    assert json.loads((tmp_path / "out/contamination.json").read_text())["ok"] is False
