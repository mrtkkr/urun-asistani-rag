import hashlib
import math

from django.conf import settings

from llm.base import ChatResponse, LLMProvider


class FakeProvider(LLMProvider):
    """Deterministic provider for tests; never touches the network.

    `embed` hashes each text into a unit vector. `chat` pops scripted
    ChatResponse objects from `responses` (or echoes the last user message).
    Set `fail_embed_times` to simulate that many consecutive rate-limit errors.
    """

    def __init__(self, responses=None, fail_embed_times=0):
        self.responses = list(responses or [])
        self.fail_embed_times = fail_embed_times
        self.embed_calls: list[list[str]] = []
        self.chat_calls: list[dict] = []

    def embed(self, texts):
        from llm.base import RateLimitError

        if self.fail_embed_times > 0:
            self.fail_embed_times -= 1
            raise RateLimitError("fake 429")
        self.embed_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def chat(self, messages, tools=None):
        self.chat_calls.append({"messages": messages, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return ChatResponse(text=f"fake: {last_user}")

    @staticmethod
    def _vector(text):
        dim = settings.EMBED_DIM
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [seed[i % len(seed)] - 127.5 + (i % 7) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
