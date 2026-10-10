import math

from django.conf import settings
from google import genai
from google.genai import errors, types

from llm.base import EMBED_TASKS, ChatResponse, LLMProvider, RateLimitError, ToolCall


def _translate(exc: errors.APIError):
    if exc.code == 429:
        raise RateLimitError(str(exc)) from exc
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
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

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

    def chat(self, messages, tools=None):
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
        )
        try:
            response = self.client.models.generate_content(
                model=settings.CHAT_MODEL,
                contents=self._contents(messages),
                config=config,
            )
        except errors.APIError as exc:
            _translate(exc)

        text_parts, calls = [], []
        parts = response.candidates[0].content.parts if response.candidates else []
        for part in parts or []:
            if part.function_call:
                fc = part.function_call
                calls.append(ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id or fc.name))
            elif part.text:
                text_parts.append(part.text)
        return ChatResponse(text="".join(text_parts), tool_calls=calls)

    @staticmethod
    def _contents(messages):
        contents = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif role == "assistant":
                parts = [types.Part(text=m["content"])] if m.get("content") else []
                parts += [
                    types.Part(function_call=types.FunctionCall(name=c["name"], args=c["args"]))
                    for c in m.get("tool_calls", [])
                ]
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                contents.append(types.Content(role="user", parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name=m["name"], response={"result": m["content"]}
                    )
                )]))
        return contents
