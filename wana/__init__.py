"""Evidence for fine-tuning dataset contamination, without runtime dependencies."""

__version__ = "0.1.0"
from wana.api import check_contamination
from wana.domain.contamination import Hit, Level, Report
from wana.domain.example import Example, Message, Role

__all__ = ["Example", "Hit", "Level", "Message", "Report", "Role", "check_contamination"]
