from django.conf import settings
from django.db import models
from pgvector.django import VectorField


class Chunk(models.Model):
    class SourceType(models.TextChoices):
        PRODUCT = "product", "Ürün"
        FAQ = "faq", "SSS"

    text = models.TextField()
    source_type = models.CharField(max_length=20, choices=SourceType.choices)
    # Product slug, or a stable slug of the FAQ question.
    source_id = models.CharField(max_length=120)
    chunk_index = models.PositiveSmallIntegerField(default=0)
    title = models.CharField(max_length=200)
    embedding = VectorField(dimensions=settings.EMBED_DIM)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["source_type", "source_id", "chunk_index"], name="unique_chunk_per_source"
            )
        ]
        ordering = ["source_type", "source_id", "chunk_index"]

    def __str__(self):
        return f"{self.source_type}:{self.source_id}#{self.chunk_index}"
