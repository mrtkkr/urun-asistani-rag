import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from catalog.models import Product

FIELDS = [
    "name", "category", "description", "ingredients", "allergens",
    "net_weight_g", "storage", "origin", "price", "stock",
]


class Command(BaseCommand):
    help = "Load data/products.json into the Product table (idempotent, keyed by slug)."

    def add_arguments(self, parser):
        parser.add_argument("--file", default=str(settings.DATA_DIR / "products.json"))

    def handle(self, *args, **options):
        try:
            with open(options["file"], encoding="utf-8") as f:
                items = json.load(f)
        except FileNotFoundError:
            raise CommandError(f"File not found: {options['file']}")

        created = updated = 0
        for item in items:
            _, was_created = Product.objects.update_or_create(
                slug=item["slug"], defaults={k: item[k] for k in FIELDS}
            )
            created += was_created
            updated += not was_created
        self.stdout.write(self.style.SUCCESS(f"Ürünler yüklendi: {created} yeni, {updated} güncellendi."))
