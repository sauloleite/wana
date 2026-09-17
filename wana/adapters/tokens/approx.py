import re


class ApproxTokenCounter:
    """Count Unicode word/punctuation units, explicitly not model tokens."""

    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))
