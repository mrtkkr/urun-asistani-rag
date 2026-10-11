import json
import re
import time

from django.conf import settings

from llm import LLMError, get_provider
from rag.search import search

from chat.prompts import NO_ANSWER, TOOL_LIMIT_ANSWER, UNAVAILABLE_MESSAGE, build_messages
from chat.tools import TOOL_DEFINITIONS, run_tool

_CITATION = re.compile(r"\[([\d,\s]+)\]")


class ChatUnavailableError(Exception):
    """The model could not be reached even after retries; str() is a user-facing Turkish message."""


def search_query(message: str, history: list[dict]) -> str:
    """Retrieval query: the last two user messages, so "peki fiyatı?" finds the earlier product (K-9)."""
    previous = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
    return f"{previous}\n{message}" if previous else message


def _with_retry(call):
    """Run call(); on 429/503/timeout wait briefly and retry (at most CHAT_MAX_RETRIES times)."""
    for attempt in range(settings.CHAT_MAX_RETRIES + 1):
        try:
            return call()
        except LLMError:
            if attempt == settings.CHAT_MAX_RETRIES:
                raise ChatUnavailableError(UNAVAILABLE_MESSAGE) from None
            time.sleep(settings.CHAT_RETRY_WAIT)


def _cited_numbers(answer: str) -> set[int]:
    return {int(n) for group in _CITATION.findall(answer) for n in re.findall(r"\d+", group)}


def _sources(results, answer: str = "") -> list[dict]:
    """One entry per source (best score wins); a product has several chunks.

    `cited` is true when the answer cites any of the source's chunk numbers ([n]).
    """
    cited = _cited_numbers(answer)
    sources = {}
    for number, r in enumerate(results, start=1):  # results arrive best first
        key = (r.chunk.source_type, r.chunk.source_id)
        entry = sources.setdefault(key, {
            "type": r.chunk.source_type,
            "id": r.chunk.source_id,
            "title": r.chunk.title,
            "score": round(r.score, 4),
            "cited": False,
        })
        entry["cited"] = entry["cited"] or number in cited
    return list(sources.values())


def _run_tools(response, messages, tool_calls) -> None:
    """Append the assistant's tool request and each tool result to the conversation."""
    messages.append({
        "role": "assistant",
        "content": response.text,
        "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in response.tool_calls],
        "raw": response.raw,
    })
    for call in response.tool_calls:
        result = run_tool(call.name, call.args)
        tool_calls.append({"name": call.name, "args": call.args, "result": result})
        messages.append({
            "role": "tool",
            "name": call.name,
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })


def answer_question(message: str, history: list[dict], provider=None, results=None) -> dict:
    """Retrieve, then let the model answer, running its tool calls for at most CHAT_MAX_TOOL_TURNS turns.

    Raises ChatUnavailableError when the model stays unreachable after retries.
    """
    provider = provider or get_provider()
    if results is None:
        results = _with_retry(lambda: search(search_query(message, history), provider=provider))
    if not results:
        # Nothing relevant: skip the model entirely rather than let it improvise.
        return {"answer": NO_ANSWER, "sources": [], "tool_calls": []}

    messages = build_messages(message, history, [r.chunk for r in results])
    tool_calls: list[dict] = []
    for _ in range(settings.CHAT_MAX_TOOL_TURNS):
        response = _with_retry(lambda: provider.chat(messages, TOOL_DEFINITIONS))
        if not response.tool_calls:
            answer = response.text.strip()
            return {"answer": answer, "sources": _sources(results, answer), "tool_calls": tool_calls}
        _run_tools(response, messages, tool_calls)
    return {"answer": TOOL_LIMIT_ANSWER, "sources": _sources(results), "tool_calls": tool_calls}
