import json
import logging
import re
import time

from django.conf import settings

from llm import LLMError, ProviderUnavailableError, get_provider
from rag.search import search

from chat.prompts import (NO_ANSWER, TOOL_LIMIT_ANSWER, UNAVAILABLE_MESSAGE, build_messages,
                          group_by_source)
from chat.tools import TOOL_DEFINITIONS, run_tool

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[([\d,\s]+)\]")


class ChatUnavailableError(Exception):
    """The model could not be reached even after retries; str() is a user-facing Turkish message."""


def search_query(message: str, history: list[dict]) -> str:
    """Retrieval query: the last two user messages, so "peki fiyatı?" finds the earlier product (K-9)."""
    previous = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
    return f"{previous}\n{message}" if previous else message


def _describe(exc: LLMError) -> str:
    """Error type and HTTP status only; never the message, which could echo request details."""
    cause = f", cause {type(exc.__cause__).__name__}" if exc.__cause__ else ""
    return f"{type(exc).__name__} (status {exc.status_code or 'none'}{cause})"


def _with_retry(call, fallback_model: str = ""):
    """Run call(); on 429/503/timeout wait briefly and retry (at most CHAT_MAX_RETRIES times).

    call receives an optional model name. When fallback_model is set and the earlier attempt
    failed with a 5xx or timeout at some point (a lone 429 does not trigger it), the last
    attempt uses it instead of the main model.
    """
    attempts = settings.CHAT_MAX_RETRIES + 1
    model = None
    saw_unavailable = False
    for attempt in range(1, attempts + 1):
        try:
            return call(model) if fallback_model else call()
        except LLMError as exc:
            saw_unavailable = saw_unavailable or isinstance(exc, ProviderUnavailableError)
            if attempt == attempts - 1 and fallback_model and saw_unavailable:
                model = fallback_model
                logger.warning("Main model unavailable (%s); last attempt uses fallback model %s",
                               _describe(exc), fallback_model)
                continue
            if attempt == attempts:
                logger.warning("LLM call failed, giving up after %d attempts: %s", attempts, _describe(exc))
                raise ChatUnavailableError(UNAVAILABLE_MESSAGE) from None
            logger.warning("LLM call failed (attempt %d/%d), retrying in %gs: %s",
                           attempt, attempts, settings.CHAT_RETRY_WAIT, _describe(exc))
            time.sleep(settings.CHAT_RETRY_WAIT)


def _cited_numbers(answer: str) -> set[int]:
    return {int(n) for group in _CITATION.findall(answer) for n in re.findall(r"\d+", group)}


def _sources(results, answer: str = "") -> list[dict]:
    """One entry per source, numbered exactly as in the prompt (`n`); best chunk score wins.

    `cited` is true when the answer cites the source's number ([n]).
    """
    cited = _cited_numbers(answer)
    sources = []
    for n, group in enumerate(group_by_source(results, lambda r: r.chunk), start=1):
        best = max(group, key=lambda r: r.score)  # results arrive best first
        sources.append({
            "n": n,
            "type": best.chunk.source_type,
            "id": best.chunk.source_id,
            "title": best.chunk.title,
            "score": round(best.score, 4),
            "cited": n in cited,
        })
    return sources


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
        response = _with_retry(lambda model=None: provider.chat(messages, TOOL_DEFINITIONS, model=model),
                               fallback_model=settings.CHAT_MODEL_FALLBACK)
        if not response.tool_calls:
            answer = response.text.strip()
            return {"answer": answer, "sources": _sources(results, answer), "tool_calls": tool_calls}
        _run_tools(response, messages, tool_calls)
    return {"answer": TOOL_LIMIT_ANSWER, "sources": _sources(results), "tool_calls": tool_calls}
