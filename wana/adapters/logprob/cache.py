"""Content-addressed NLL cache including provider identity and prompt messages."""

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path

from wana.domain.example import Message
from wana.ports.logprob import LogProbProvider


class CachedLogProbProvider:
    def __init__(self, provider: LogProbProvider, path: Path, identity: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.provider, self.identity = provider, identity
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS nll (key TEXT PRIMARY KEY, value REAL NOT NULL)"
        )

    def mean_nll(self, prefix: tuple[Message, ...], target: str) -> float:
        payload = json.dumps([self.identity, [asdict(m) for m in prefix], target], sort_keys=True)
        key = hashlib.sha256(payload.encode()).hexdigest()
        row = self.connection.execute("SELECT value FROM nll WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return float(row[0])
        value = self.provider.mean_nll(prefix, target)
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid NLL for cache")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO nll VALUES (?, ?)", (key, value))
        return value

    def close(self) -> None:
        self.connection.close()
