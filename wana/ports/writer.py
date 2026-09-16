from typing import Protocol

from wana.domain.contamination import Report


class ReportWriter(Protocol):
    def write(self, report: Report, path: str) -> None: ...
