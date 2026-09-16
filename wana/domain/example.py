"""Format-independent, immutable training examples."""

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Example:
    id: str
    messages: tuple[Message, ...]

    @property
    def text(self) -> str:
        """Return message contents in conversation order."""
        return "\n".join(message.content for message in self.messages)
