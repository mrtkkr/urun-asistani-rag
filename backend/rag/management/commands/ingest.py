from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from llm import get_provider
from rag.ingest import ingest


class Command(BaseCommand):
    help = "Chunk products and FAQ, embed them in batches and store them in pgvector (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument("--batch-size", type=int, default=settings.INGEST_BATCH_SIZE)
        parser.add_argument("--pause", type=float, default=settings.INGEST_BATCH_PAUSE,
                            help="Seconds to wait between batches.")

        parser.add_argument("--force", action="store_true",
                            help="Re-embed every chunk (after changing the embedding model or format).")

    def handle(self, *args, **options):
        try:
            stats = ingest(
                get_provider(),
                batch_size=options["batch_size"],
                pause=options["pause"],
                max_retries=settings.INGEST_MAX_RETRIES,
                wait=settings.INGEST_RETRY_WAIT,
                log=self.stdout.write,
                force=options["force"],
            )
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            raise CommandError(str(exc))
        self.stdout.write(self.style.SUCCESS(
            "Bitti: {created} yeni, {updated} güncellendi, {unchanged} değişmedi, {deleted} silindi.".format(**stats)
        ))
