"""Turn catalog rows and the FAQ file into embeddable text chunks.

Price and stock are deliberately absent: they change, so the model must read
them through tool calls (K-5).
"""
import re
from dataclasses import dataclass

from django.utils.text import slugify

from catalog.models import Product
from rag.models import Chunk


@dataclass(frozen=True)
class ChunkSpec:
    source_type: str
    source_id: str
    chunk_index: int
    title: str
    text: str

    @property
    def key(self):
        return (self.source_type, self.source_id, self.chunk_index)


def product_chunks(product: Product) -> list[ChunkSpec]:
    allergens = ", ".join(product.allergens) if product.allergens else "yok"
    texts = [
        f"{product.name} ({product.category}). {product.description}",
        (
            f"{product.name} ürün bilgileri. İçindekiler: {product.ingredients} "
            f"Alerjenler: {allergens}. Net ağırlık: {product.net_weight_g} g. "
            f"Saklama koşulu: {product.storage}. Menşe: {product.origin}."
        ),
    ]
    return [
        ChunkSpec(Chunk.SourceType.PRODUCT, product.slug, i, product.name, text)
        for i, text in enumerate(texts)
    ]


def faq_chunks(markdown: str) -> list[ChunkSpec]:
    """Each '## Question' heading plus the answer below it is one chunk."""
    specs = []
    for section in re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]:
        question, _, answer = section.partition("\n")
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            continue
        specs.append(ChunkSpec(
            Chunk.SourceType.FAQ,
            slugify(question, allow_unicode=False)[:120] or f"faq-{len(specs)}",
            0,
            question,
            f"Soru: {question}\nCevap: {answer}",
        ))
    return specs
