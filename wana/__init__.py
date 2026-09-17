"""Auditable fine-tuning curation with a bundled offline scoring model."""

__version__ = "0.5.0"
from wana.adapters.io.provenance import verify_manifest
from wana.api import check_contamination, score_dataset, select_subset
from wana.domain.contamination import Hit, Level, Report
from wana.domain.example import Example, Message, Role
from wana.workflows import run

__all__ = [
    "Example",
    "Hit",
    "Level",
    "Message",
    "Report",
    "Role",
    "check_contamination",
    "score_dataset",
    "select_subset",
    "run",
    "verify_manifest",
]
