from dataclasses import dataclass

from django.conf import settings
from pgvector.django import CosineDistance

from llm import get_provider
from rag.models import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity: 1 - cosine distance


def search(query: str, k: int | None = None, min_score: float | None = None,
           provider=None) -> list[SearchResult]:
    """Return the k chunks closest to the query, best first, dropping weak matches."""
    k = settings.SEARCH_TOP_K if k is None else k
    min_score = settings.SEARCH_MIN_SCORE if min_score is None else min_score
    provider = provider or get_provider()
    [vector] = provider.embed([query], task="query")
    rows = Chunk.objects.annotate(distance=CosineDistance("embedding", vector)).order_by("distance")[:k]
    results = [SearchResult(chunk=row, score=1.0 - row.distance) for row in rows]
    return [r for r in results if r.score >= min_score]
