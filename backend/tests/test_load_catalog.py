import json

import pytest
from django.core.management import call_command

from catalog.models import Product


@pytest.mark.django_db
def test_load_catalog_loads_all_products(catalog):
    assert 25 <= Product.objects.count() <= 35
    cheese = Product.objects.get(slug="beyaz-peynir-tam-yagli")
    assert cheese.allergens == ["süt"]
    assert cheese.price > 0


@pytest.mark.django_db
def test_load_catalog_is_idempotent_and_updates(catalog, tmp_path):
    count = Product.objects.count()
    call_command("load_catalog")
    assert Product.objects.count() == count

    path = tmp_path / "p.json"
    first = Product.objects.get(slug="kasar-eski")
    path.write_text(json.dumps([{
        "slug": first.slug, "name": first.name, "category": first.category,
        "description": first.description, "ingredients": first.ingredients,
        "allergens": first.allergens, "net_weight_g": first.net_weight_g,
        "storage": first.storage, "origin": first.origin, "price": "1.00", "stock": 7,
    }]), encoding="utf-8")
    call_command("load_catalog", file=str(path))
    first.refresh_from_db()
    assert first.stock == 7 and str(first.price) == "1.00"
