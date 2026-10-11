import math

import httpx

from django.conf import settings
from google import genai
from google.genai import errors, types

from llm.base import (EMBED_TASKS, ChatResponse, LLMProvider, ProviderUnavailableError,
                      RateLimitError, ToolCall)


def _translate(exc: errors.APIError):
    if exc.code == 429:
        raise RateLimitError(str(exc), status_code=exc.code) from exc
    if exc.code in (500, 502, 503, 504):
        raise ProviderUnavailableError(str(exc), status_code=exc.code) from exc
    raise exc


def _normalize(vector: list[float]) -> list[float]:
    # gemini-embedding-001 returns non-unit vectors below 3072 dims (gemini-embedding-2
    # already normalizes; doing it again is a harmless no-op), so always L2-normalize.
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class GeminiProvider(LLMProvider):
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example).")
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=int(settings.LLM_TIMEOUT_S * 1000)),
        )

    def embed(self, texts, task="document", titles=None):
        texts = list(texts)
        if not texts:
            return []
        texts = self._with_task_format(texts, task, titles)
        # gemini-embedding-2 folds a plain list of strings into ONE aggregated embedding.
        # Wrapping each text in its own Content yields one embedding per entry, in order.
        try:
            result = self.client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
                config=types.EmbedContentConfig(output_dimensionality=settings.EMBED_DIM),
            )
        except errors.APIError as exc:
            _translate(exc)
        vectors = [_normalize(list(e.values)) for e in result.embeddings or []]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: sent {len(texts)} texts, got {len(vectors)} vectors."
            )
        return vectors

    @staticmethod
    def _with_task_format(texts, task, titles):
        """gemini-embedding-2 has no task_type; the task is a text prefix (K-6).

        Format from the official embeddings guide: queries are
        "task: search result | query: ...", documents "title: ... | text: ...".
        """
        if task not in EMBED_TASKS:
            raise ValueError(f"Unknown embed task: {task!r}")
        if task == "query":
            return [f"task: search result | query: {t}" for t in texts]
        titles = titles or ["none"] * len(texts)
        if len(titles) != len(texts):
            raise ValueError("titles must have one entry per text.")
        return [f"title: {title or 'none'} | text: {t}" for title, t in zip(titles, texts)]

    @staticmethod
    def _thinking_config(model: str):
        """Thinking level for Gemini 3 models (thinking_level); None when unset or unsupported.

        Older model families use a different parameter (thinking_budget), so the level is
        simply not sent to them.
        """
        level = (settings.CHAT_THINKING_LEVEL or "").strip().lower()
        if not level or not model.startswith("gemini-3"):
            return None
        return types.ThinkingConfig(thinking_level=level)

    def chat(self, messages, tools=None, model=None):
        model = model or settings.CHAT_MODEL
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters_json_schema=t["parameters"],
                )
                for t in tools
            ])] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=self._thinking_config(model),
        )
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=self._contents(messages),
                config=config,
            )
        except errors.APIError as exc:
            _translate(exc)
        except httpx.TransportError as exc:  # timeouts, connection resets
            raise ProviderUnavailableError(str(exc)) from exc

        text_parts, calls = [], []
        content = response.candidates[0].content if response.candidates else None
        for part in (content.parts if content else None) or []:
            if part.function_call:
                fc = part.function_call
                calls.append(ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id or ""))
            elif part.text and not getattr(part, "thought", None):
                text_parts.append(part.text)
        # Keep the model's own Content: Gemini 3 thought signatures on function_call parts must
        # be sent back unchanged in the next request of the tool loop.
        return ChatResponse(text="".join(text_parts), tool_calls=calls, raw=content if calls else None)

    @staticmethod
    def _contents(messages):
        contents = []
        after_tool = False  # previous message was a tool result
        for m in messages:
            role = m["role"]
            if role == "system":
                continue
            was_tool, after_tool = after_tool, role == "tool"
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif role == "assistant":
                if m.get("raw") is not None:
                    contents.append(m["raw"])
                    continue
                parts = [types.Part(text=m["content"])] if m.get("content") else []
                parts += [
                    types.Part(function_call=types.FunctionCall(name=c["name"], args=c["args"]))
                    for c in m.get("tool_calls", [])
                ]
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                part = types.Part(function_response=types.FunctionResponse(
                    id=m.get("tool_call_id") or None, name=m["name"], response={"result": m["content"]}
                ))
                # Results of parallel calls belong in one Content, in call order.
                if was_tool:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
        return contents
