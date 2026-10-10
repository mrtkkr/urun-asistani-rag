from django.conf import settings

from llm.base import ChatResponse, LLMProvider, RateLimitError, ToolCall

__all__ = ["ChatResponse", "LLMProvider", "RateLimitError", "ToolCall", "get_provider"]


def get_provider() -> LLMProvider:
    """Build the provider selected by LLM_PROVIDER (imports are lazy)."""
    name = settings.LLM_PROVIDER.lower()
    if name == "gemini":
        from llm.gemini import GeminiProvider
        return GeminiProvider()
    if name == "fake":
        from llm.fake import FakeProvider
        return FakeProvider()
    if name == "ollama":
        raise NotImplementedError("The Ollama provider is planned for Round 5.")
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER!r}")
