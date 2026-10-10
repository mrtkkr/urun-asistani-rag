from django.core.management.base import BaseCommand, CommandError

from llm import get_provider
from llm.base import RateLimitError
from rag.search import search

from chat.service import answer_question


class Command(BaseCommand):
    help = "Ask the assistant a question from the terminal and show the answer, sources and scores."

    def add_arguments(self, parser):
        parser.add_argument("question")
        parser.add_argument("--raw", action="store_true",
                            help="Only list the closest chunks with scores, ignoring SEARCH_MIN_SCORE "
                                 "(for tuning the threshold); the model is not called.")
        parser.add_argument("-k", type=int, default=None, help="Number of chunks to retrieve.")

    def handle(self, *args, **options):
        try:
            provider = get_provider()
            if options["raw"]:
                for r in search(options["question"], k=options["k"] or 10, min_score=-1.0,
                                provider=provider):
                    self.stdout.write(f"{r.score:.4f}  {r.chunk.source_type}:{r.chunk.source_id}  {r.chunk.title}")
                return
            results = search(options["question"], k=options["k"], provider=provider)
            out = answer_question(options["question"], [], provider=provider, results=results)
        except (RuntimeError, ValueError, NotImplementedError, RateLimitError) as exc:
            raise CommandError(str(exc))
        self.stdout.write(f"\nYanıt:\n{out['answer']}\n")
        self.stdout.write("Kaynaklar:" if out["sources"] else "Kaynak yok (eşiği geçen parça bulunamadı).")
        for s in out["sources"]:
            self.stdout.write(f"  {s['score']:.4f}  {s['type']}:{s['id']}  {s['title']}")
