import pytest
from django.conf import settings
from django.db import IntegrityError

from catalog.models import Product
from rag.models import Chunk


def make_product(**overrides):
    data = dict(
        slug="test-peynir", name="Test Peynir", category="Peynir", description="Açıklama",
        ingredients="süt, tuz", allergens=["süt"], net_weight_g=250, storage="Buzdolabı",
        origin="Kars", price="99.90", stock=5,
    )
    return Product.objects.create(**{**data, **overrides})


@pytest.mark.django_db
def test_product_roundtrip():
    product = make_product()
    product.refresh_from_db()
    assert str(product) == "Test Peynir"
    assert product.allergens == ["süt"]
    assert str(product.price) == "99.90"


@pytest.mark.django_db
def test_product_slug_is_unique():
    make_product()
    with pytest.raises(IntegrityError):
        make_product()


@pytest.mark.django_db
def test_chunk_stores_vector_of_embed_dim():
    vector = [0.1] * settings.EMBED_DIM
    Chunk.objects.create(text="t", source_type="faq", source_id="x", title="x", embedding=vector)
    assert len(Chunk.objects.get().embedding) == settings.EMBED_DIM


@pytest.mark.django_db
def test_chunk_source_is_unique_per_index():
    vector = [0.0] * settings.EMBED_DIM
    kwargs = dict(text="t", source_type="faq", source_id="x", chunk_index=0, title="x", embedding=vector)
    Chunk.objects.create(**kwargs)
    with pytest.raises(IntegrityError):
        Chunk.objects.create(**kwargs)
