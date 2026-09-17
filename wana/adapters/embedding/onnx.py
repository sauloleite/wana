"""Local ONNX sentence encoder with attention-mask mean pooling."""

import importlib
from pathlib import Path

from wana.domain.vectors import cosine


class OnnxEmbedder:
    def __init__(self, model: str, tokenizer: str, *, max_tokens: int = 512) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        try:
            ort = importlib.import_module("onnxruntime")
            tokenizers = importlib.import_module("tokenizers")
            self.np = importlib.import_module("numpy")
        except ImportError as exc:
            raise ValueError("install wana[embed] for ONNX embeddings") from exc
        self.session = ort.InferenceSession(str(Path(model)), providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer)
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.inputs = tuple(item.name for item in self.session.get_inputs())
        if any(
            name not in ("input_ids", "attention_mask", "token_type_ids") for name in self.inputs
        ):
            raise ValueError("unsupported ONNX input names")

    def embed(self, text: str) -> tuple[float, ...]:
        encoded = self.tokenizer.encode(text)
        fields = {
            "input_ids": encoded.ids,
            "attention_mask": encoded.attention_mask,
            "token_type_ids": encoded.type_ids,
        }
        feed = {name: self.np.asarray([fields[name]], dtype=self.np.int64) for name in self.inputs}
        output = self.session.run(None, feed)[0]
        if output.ndim == 3:
            mask = self.np.asarray(encoded.attention_mask, dtype=self.np.float32)
            vector = (output[0] * mask[:, None]).sum(axis=0) / max(1.0, float(mask.sum()))
        elif output.ndim == 2:
            vector = output[0]
        else:
            raise ValueError("ONNX output must be [batch, tokens, dimension] or [batch, dimension]")
        result = tuple(float(value) for value in vector)
        cosine(result, result)
        return result
