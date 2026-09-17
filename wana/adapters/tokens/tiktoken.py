import importlib


class TiktokenCounter:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            self.encoding = importlib.import_module("tiktoken").get_encoding(encoding)
        except ImportError as exc:
            raise ValueError("install wana[tokens] to use tiktoken") from exc

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))
