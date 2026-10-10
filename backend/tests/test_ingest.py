import re

import pytest
from django.conf import settings
from django.core.management import call_command

from catalog.models import Product
from llm.fake import FakeProvider
from rag.chunking import faq_chunks, product_chunks
from rag.ingest import ingest
from rag.models import Chunk

KW = dict(batch_size=10, sleep=lambda s: None)


def test_faq_chunks_split_by_heading():
    specs = faq_chunks("# T\n\n## Soru bir?\nCevap bir.\n\n## Soru iki?\nCevap iki.\n")
    assert [s.title for s in specs] == ["Soru bir?", "Soru iki?"]
    assert "Cevap bir." in specs[0].text


@pytest.mark.django_db
def test_product_chunks_exclude_price_and_stock(catalog):
    product = Product.objects.get(slug="beyaz-peynir-tam-yagli")
    text = " ".join(s.text for s in product_chunks(product))
    assert str(product.price) not in text
    assert "189" not in text
    assert not re.search(r"stok|fiyat", text, re.IGNORECASE)


@pytest.mark.django_db
def test_ingest_creates_chunks_with_fake_provider(catalog):
    provider = FakeProvider()
    stats = ingest(provider, **KW)
    assert stats["created"] == Chunk.objects.count() > 0
    assert Chunk.objects.filter(source_type="product").count() == Product.objects.count() * 2
    assert Chunk.objects.filter(source_type="faq").count() >= 10
    assert all(len(c.embedding) == settings.EMBED_DIM for c in Chunk.objects.all())
    assert max(len(batch) for batch in provider.embed_calls) <= 10


@pytest.mark.django_db
def test_ingest_twice_does_not_duplicate_or_reembed(catalog):
    ingest(FakeProvider(), **KW)
    count = Chunk.objects.count()
    provider = FakeProvider()
    stats = ingest(provider, **KW)
    assert Chunk.objects.count() == count
    assert stats["created"] == stats["updated"] == 0
    assert stats["unchanged"] == count
    assert provider.embed_calls == []


@pytest.mark.django_db
def test_ingest_reembeds_changed_and_deletes_stale(catalog):
    ingest(FakeProvider(), **KW)
    Product.objects.filter(slug="kasar-eski").update(description="Yepyeni açıklama.")
    Product.objects.filter(slug="pastirma").delete()
    provider = FakeProvider()
    stats = ingest(provider, **KW)
    assert stats["updated"] == 1 and stats["deleted"] == 2
    assert not Chunk.objects.filter(source_id="pastirma").exists()
    assert "Yepyeni açıklama." in Chunk.objects.get(source_id="kasar-eski", chunk_index=0).text


@pytest.mark.django_db
def test_ingest_retries_after_rate_limit(catalog):
    waits = []
    provider = FakeProvider(fail_embed_times=2)
    ingest(provider, batch_size=100, wait=5, sleep=waits.append)
    assert waits == [5, 10]
    assert Chunk.objects.count() > 0


@pytest.mark.django_db
def test_ingest_gives_up_after_max_retries(catalog):
    from llm.base import RateLimitError

    with pytest.raises(RateLimitError):
        ingest(FakeProvider(fail_embed_times=99), batch_size=100, max_retries=2, **{"sleep": lambda s: None})
    assert Chunk.objects.count() == 0


@pytest.mark.django_db
def test_ingest_rejects_wrong_dimension(catalog, settings):
    provider = FakeProvider()
    settings.EMBED_DIM = 8  # provider vectors are built with this size ...
    provider.embed = lambda texts: [[0.0] * 4 for _ in texts]  # ... but returns 4
    with pytest.raises(ValueError, match=r"expected .* size 8, got .* size\(s\) \[4\]"):
        ingest(provider, **KW)


@pytest.mark.django_db
def test_ingest_command_uses_configured_provider(catalog, capsys):
    call_command("ingest", pause=0)
    assert Chunk.objects.count() > 0
    assert "Bitti" in capsys.readouterr().out
