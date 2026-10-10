from llm import get_provider
from rag.search import search

from chat.prompts import NO_ANSWER, build_messages


def _sources(results) -> list[dict]:
    """One entry per source (best score wins); a product has several chunks."""
    best = {}
    for r in results:  # results arrive best first
        key = (r.chunk.source_type, r.chunk.source_id)
        best.setdefault(key, {
            "type": r.chunk.source_type,
            "id": r.chunk.source_id,
            "title": r.chunk.title,
            "score": round(r.score, 4),
        })
    return list(best.values())


def answer_question(message: str, history: list[dict], provider=None, results=None) -> dict:
    provider = provider or get_provider()
    if results is None:
        results = search(message, provider=provider)
    if not results:
        # Nothing relevant: skip the model entirely rather than let it improvise.
        return {"answer": NO_ANSWER, "sources": [], "tool_calls": []}
    messages = build_messages(message, history, [r.chunk for r in results])
    response = provider.chat(messages)
    return {"answer": response.text.strip(), "sources": _sources(results), "tool_calls": []}
