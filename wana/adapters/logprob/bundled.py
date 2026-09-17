"""Offline IFD from SmolLM2 weights installed by the normal pip dependency."""

import hashlib
import importlib
import importlib.metadata
from pathlib import Path

from wana.domain.example import Message


class BundledLogProbProvider:
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct:Q4_1"

    def __init__(self, *, max_tokens: int = 2048, threads: int = 4) -> None:
        if max_tokens < 2 or threads < 1:
            raise ValueError("positive threads and max_tokens >= 2 required")
        try:
            distribution = importlib.metadata.distribution("llm-smollm2")
            llama = importlib.import_module("llama_cpp")
            self.np = importlib.import_module("numpy")
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise ValueError("install wana with its dependencies for the bundled scorer") from exc
        path = Path(str(distribution.locate_file("llm_smollm2/SmolLM2-135M-Instruct.Q4_1.gguf")))
        self.revision = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model = llama.Llama(
            model_path=str(path),
            n_ctx=max_tokens,
            logits_all=True,
            n_gpu_layers=0,
            n_threads=threads,
            verbose=False,
            seed=42,
        )
        self.max_tokens = max_tokens
        self.chat_template = self.model.metadata.get("tokenizer.chat_template")
        if not self.chat_template:
            raise ValueError("bundled model does not provide a chat template")
        sandbox = importlib.import_module("jinja2.sandbox")
        self.template = sandbox.ImmutableSandboxedEnvironment().from_string(self.chat_template)

    def mean_nll(self, prefix: tuple[Message, ...], target: str) -> float:
        if not target.strip():
            raise ValueError("target must not be empty")
        if prefix:
            prompt = self.template.render(
                messages=[{"role": m.role.value, "content": m.content} for m in prefix],
                add_generation_prompt=True,
            )
            context = self.model.tokenize(prompt.encode(), add_bos=False, special=True)
        else:
            start = self.model.token_bos()
            context = [start if start >= 0 else self.model.token_eos()]
        answer = self.model.tokenize(target.encode(), add_bos=False, special=False)
        tokens = context + answer
        if not context or not answer or len(tokens) > self.max_tokens:
            raise ValueError(
                "empty token span or sequence exceeds max_tokens; no silent truncation"
            )
        self.model.reset()
        self.model.eval(tokens)
        # Row t predicts token t+1. Mask the context and retain all target tokens.
        logits = self.model.scores[len(context) - 1 : len(tokens) - 1].astype(self.np.float64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        losses = (
            self.np.log(self.np.exp(shifted).sum(axis=1))
            - shifted[self.np.arange(len(answer)), answer]
        )
        return float(losses.mean())
