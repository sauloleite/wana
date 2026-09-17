"""Real runtime smoke tests: no model download during tests."""

import importlib.util
import math
import os
from pathlib import Path

import pytest

from wana.adapters.logprob.bundled import BundledLogProbProvider
from wana.domain.example import Message, Role


def test_bundled_model_is_installed_and_offline():
    if importlib.util.find_spec("llama_cpp") is None:
        pytest.skip("model runtime not installed")
    provider = BundledLogProbProvider(max_tokens=128, threads=1)
    prefix = (Message(Role.USER, "What is two plus two?"),)
    conditional = provider.mean_nll(prefix, "Four.")
    unconditional = provider.mean_nll((), "Four.")
    assert math.isfinite(conditional) and conditional > 0
    assert math.isfinite(unconditional) and unconditional > 0
    assert provider.mean_nll(prefix, "Four.") == pytest.approx(conditional, abs=1e-6)
    with pytest.raises(ValueError):
        provider.mean_nll(prefix, "")
    with pytest.raises(ValueError):
        provider.mean_nll(prefix, "long " * 300)


def test_transformers_cpu_local_model():
    path = os.environ.get("WANA_TEST_TRANSFORMERS_MODEL")
    if not path:
        pytest.skip("set WANA_TEST_TRANSFORMERS_MODEL to a local snapshot")
    from wana.adapters.logprob.transformers_cpu import TransformersCPU

    provider = TransformersCPU(path, max_tokens=128)
    prefix = (Message(Role.USER, "What is two plus two?"),)
    assert provider.mean_nll(prefix, "Four.") > 0
    assert provider.mean_nll((), "Four.") > 0
    with pytest.raises(ValueError):
        provider.mean_nll(prefix, "long " * 300)


def test_onnx_real_local_model():
    path = os.environ.get("WANA_TEST_ONNX_MODEL")
    if not path:
        pytest.skip("set WANA_TEST_ONNX_MODEL to a local model directory")
    from wana.adapters.embedding.onnx import OnnxEmbedder
    from wana.domain.vectors import cosine

    provider = OnnxEmbedder(
        str(Path(path) / "onnx/model_quantized.onnx"), str(Path(path) / "tokenizer.json")
    )
    first = provider.embed("The cat is sleeping on the sofa.")
    paraphrase = provider.embed("A feline is resting on the couch.")
    unrelated = provider.embed("The database index needs rebuilding.")
    assert len(first) == 384
    assert cosine(first, paraphrase) > cosine(first, unrelated)
