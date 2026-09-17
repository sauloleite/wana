"""Lazy CPU causal LM adapter; only target tokens contribute to mean loss."""

import hashlib
import importlib
import math
from pathlib import Path

from wana.domain.example import Message


class TransformersCPU:
    def __init__(self, model: str, *, revision: str | None = None, max_tokens: int = 4096) -> None:
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least two")
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise ValueError("install wana[score] for transformers_cpu") from exc
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model, revision=revision)
        self.model.to("cpu")
        self.model.eval()
        self.max_tokens = max_tokens
        self.model_name = model
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        if Path(model).is_dir():
            fingerprint = hashlib.sha256()
            for path in sorted(Path(model).iterdir()):
                if path.is_file() and path.suffix in (".json", ".safetensors", ".bin"):
                    fingerprint.update(path.name.encode())
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            fingerprint.update(chunk)
            self.revision = "local-sha256:" + fingerprint.hexdigest()
        self.chat_template = self.tokenizer.chat_template

    def mean_nll(self, prefix: tuple[Message, ...], target: str) -> float:
        if not target.strip():
            raise ValueError("target must not be empty")
        if prefix:
            if not self.chat_template:
                raise ValueError("scoring model must provide a chat template")
            prompt = self.tokenizer.apply_chat_template(
                [{"role": m.role.value, "content": m.content} for m in prefix],
                tokenize=False,
                add_generation_prompt=True,
            )
            context = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            start = self.tokenizer.bos_token_id
            if start is None:
                start = self.tokenizer.eos_token_id
            if start is None:
                raise ValueError("model needs a BOS or EOS token for unconditional loss")
            context = [start]
        # Tokenize target identically in both passes, with no EOS in the scored span.
        answer = self.tokenizer.encode(target, add_special_tokens=False)
        if not context or not answer or len(context) + len(answer) > self.max_tokens:
            raise ValueError(
                "empty token span or sequence exceeds max_tokens; no silent truncation"
            )
        ids = self.torch.tensor([context + answer], dtype=self.torch.long)
        labels = ids.clone()
        labels[:, : len(context)] = -100
        with self.torch.inference_mode():
            loss = float(self.model(input_ids=ids, labels=labels).loss.item())
        if not math.isfinite(loss):
            raise ValueError("model returned nonfinite loss")
        return loss
