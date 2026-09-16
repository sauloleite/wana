import json
from dataclasses import asdict
from pathlib import Path

from wana.domain.contamination import Report
from wana.domain.manifest import Manifest


def serialize(value: Report | Manifest) -> str:
    """Serialize dataclasses to reproducible UTF-8 JSON text."""
    data = asdict(value)
    if isinstance(value, Report):
        data["ok"] = value.ok
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


class JsonReportWriter:
    def write(self, report: Report, path: str) -> None:
        Path(path).write_text(serialize(report), encoding="utf-8", newline="\n")
