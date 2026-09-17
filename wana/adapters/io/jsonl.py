"""Strict, streaming JSONL input with standard-library compression."""

import bz2
import copy
import gzip
import json
import lzma
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO, cast

from wana.domain.example import Example, Message, Role


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return cast(TextIO, gzip.open(path, "rt", encoding="utf-8"))
    if path.suffix == ".xz":
        return cast(TextIO, lzma.open(path, "rt", encoding="utf-8"))
    if path.suffix == ".bz2":
        return cast(TextIO, bz2.open(path, "rt", encoding="utf-8"))
    return path.open(encoding="utf-8")


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def parse_record(record: Any, example_id: str) -> Example:
    """Normalize a supported record, rejecting silent content loss."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    messages: list[Message] = []
    if "messages" in record or "conversations" in record:
        sharegpt = "messages" not in record
        rows = record["conversations" if sharegpt else "messages"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("conversation must be a nonempty list")
        aliases = {"human": "user", "gpt": "assistant"}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("message must be an object")
            role = _string(row.get("from" if sharegpt else "role"), "role")
            content = _string(row.get("value" if sharegpt else "content"), "content")
            messages.append(Message(Role(aliases.get(role, role)), content))
    elif "instruction" in record and "output" in record:
        instruction = _string(record["instruction"], "instruction")
        context = _string(record.get("input", ""), "input")
        messages = [
            Message(Role.USER, instruction + ("\n" + context if context else "")),
            Message(Role.ASSISTANT, _string(record["output"], "output")),
        ]
    else:
        raise ValueError("expected openai-chat, alpaca or sharegpt record")
    return Example(
        example_id,
        tuple(messages),
        {
            k: copy.deepcopy(v)
            for k, v in record.items()
            if k not in ("messages", "conversations", "wana")
        },
        copy.deepcopy(record),
    )


class JsonlReader:
    def read(self, path: str) -> Iterator[Example]:
        """Yield nonblank records with path:line IDs and contextual errors."""
        with open_text(Path(path)) as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield parse_record(json.loads(line), f"{path}:{number}")
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
