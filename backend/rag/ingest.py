import time

from django.conf import settings

from catalog.models import Product
from llm.base import LLMProvider, RateLimitError
from rag.chunking import ChunkSpec, faq_chunks, product_chunks
from rag.models import Chunk


def collect_chunks(faq_path=None) -> list[ChunkSpec]:
    specs = []
    for product in Product.objects.all():
        specs += product_chunks(product)
    faq_path = faq_path or settings.DATA_DIR / "faq.md"
    if faq_path.exists():
        specs += faq_chunks(faq_path.read_text(encoding="utf-8"))
    return specs


def embed_with_retry(provider: LLMProvider, texts, max_retries, wait, sleep=time.sleep):
    """Embed one batch; on a rate-limit error wait (linear backoff) and retry."""
    for attempt in range(max_retries + 1):
        try:
            return provider.embed(texts)
        except RateLimitError:
            if attempt == max_retries:
                raise
            sleep(wait * (attempt + 1))


def ingest(provider, *, batch_size, pause=0.0, max_retries=5, wait=15.0, faq_path=None,
           sleep=time.sleep, log=lambda msg: None) -> dict:
    """Sync the Chunk table with the catalog and FAQ.

    Safe to re-run: only new or changed chunks are embedded, unchanged ones are
    skipped and chunks whose source disappeared are deleted.
    """
    desired = {s.key: s for s in collect_chunks(faq_path)}
    existing = {(c.source_type, c.source_id, c.chunk_index): c for c in Chunk.objects.all()}

    stale = [c.pk for key, c in existing.items() if key not in desired]
    Chunk.objects.filter(pk__in=stale).delete()

    todo = [s for key, s in desired.items() if key not in existing or existing[key].text != s.text]
    stats = {"created": 0, "updated": 0, "unchanged": len(desired) - len(todo), "deleted": len(stale)}

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        vectors = embed_with_retry(provider, [s.text for s in batch], max_retries, wait, sleep)
        got_dims = sorted({len(v) for v in vectors})
        if len(vectors) != len(batch) or got_dims != [settings.EMBED_DIM]:
            raise ValueError(
                f"Provider returned embeddings that do not match EMBED_DIM: expected "
                f"{len(batch)} vectors of size {settings.EMBED_DIM}, got {len(vectors)} "
                f"vectors of size(s) {got_dims}. "
                "Check EMBED_MODEL / EMBED_DIM and re-create the migration if the size changed."
            )
        for spec, vector in zip(batch, vectors):
            _, created = Chunk.objects.update_or_create(
                source_type=spec.source_type, source_id=spec.source_id, chunk_index=spec.chunk_index,
                defaults={"text": spec.text, "title": spec.title, "embedding": vector},
            )
            stats["created" if created else "updated"] += 1
        log(f"{min(start + batch_size, len(todo))}/{len(todo)} parça işlendi")
        if pause and start + batch_size < len(todo):
            sleep(pause)
    return stats
