"""Provider-independent LLM interface.

Messages are plain dicts: {"role": "system"|"user"|"assistant"|"tool", "content": str}.
An assistant message may carry "tool_calls": [{"id", "name", "args"}] and "raw"; a tool
message carries "name" and "tool_call_id" next to its "content".
Tools are dicts: {"name", "description", "parameters"} with a JSON Schema.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


EMBED_TASKS = ("document", "query")


class LLMError(Exception):
    """Base class for provider failures a caller may retry.

    status_code is the provider's HTTP status when there was one (None for timeouts etc.).
    """

    def __init__(self, message: str = "", status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(LLMError):
    """The provider answered HTTP 429; the caller may wait and retry."""


class ProviderUnavailableError(LLMError):
    """The provider answered 5xx (e.g. 503 UNAVAILABLE) or timed out; retrying may help."""


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = ""


@dataclass
class ChatResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Opaque provider payload (e.g. Gemini's model Content with thought signatures). The caller
    # puts it back on the assistant message as "raw" so the provider can replay it verbatim.
    raw: Any = None


class LLMProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str], task: str = "document",
              titles: list[str] | None = None) -> list[list[float]]:
        """Return one vector of length EMBED_DIM per input text.

        task is "document" (stored passages) or "query" (a user question); retrieval
        quality depends on embedding both sides the way the model expects (K-6).
        titles optionally gives one title per document text.
        """

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        """Run one model turn; the model may answer with tool calls instead of text."""
