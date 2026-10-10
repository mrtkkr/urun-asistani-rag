from django.db import models


class Product(models.Model):
    slug = models.SlugField(max_length=80, unique=True)
    name = models.CharField(max_length=120)
    category = models.CharField(max_length=60)
    description = models.TextField()
    ingredients = models.TextField()
    allergens = models.JSONField(default=list, blank=True)
    net_weight_g = models.PositiveIntegerField()
    storage = models.CharField(max_length=200)
    origin = models.CharField(max_length=80)
    # Volatile fields: read through tool calls only, never embedded (K-5).
    price = models.DecimalField(max_digits=8, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name
