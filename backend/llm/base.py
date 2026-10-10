"""Provider-independent LLM interface.

Messages are plain dicts: {"role": "system"|"user"|"assistant"|"tool", "content": str}.
An assistant message may carry "tool_calls": [{"id", "name", "args"}]; a tool
message carries "name" and "tool_call_id" next to its "content".
Tools are dicts: {"name", "description", "parameters"} with a JSON Schema.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class RateLimitError(Exception):
    """The provider answered HTTP 429; the caller may wait and retry."""


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = ""


@dataclass
class ChatResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector of length EMBED_DIM per input text."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        """Run one model turn; the model may answer with tool calls instead of text."""
