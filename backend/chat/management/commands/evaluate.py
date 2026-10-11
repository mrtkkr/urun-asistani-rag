from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from chat import evaluation
from llm import get_provider


class Command(BaseCommand):
    help = ("Measure retrieval (hit@1, hit@4, threshold filtering) over data/eval.json; "
            "--with-llm also measures tool calls and citations. Writes docs/eval-results.md "
            "and updates the table in README.md.")

    def add_arguments(self, parser):
        parser.add_argument("--with-llm", action="store_true",
                            help="Also call the chat model (uses free-tier quota).")
        parser.add_argument("--pause", type=float, default=8.0,
                            help="Seconds to wait between model calls with --with-llm (default 8).")
        parser.add_argument("--file", default=str(settings.DATA_DIR / "eval.json"))
        parser.add_argument("--docs-dir", default=str(settings.DOCS_DIR))
        parser.add_argument("--readme", default=str(settings.README_PATH))

    def handle(self, *args, **options):
        with_llm = options["with_llm"]
        try:
            cases = evaluation.load_cases(options["file"])
            provider = get_provider()
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}")

        results = evaluation.run(cases, provider, with_llm=with_llm, pause=options["pause"],
                                 log=lambda msg: self.stderr.write(msg))
        self.stdout.write("\n" + evaluation.terminal_report(results, with_llm) + "\n")

        docs_dir = options["docs_dir"]
        try:
            out = Path(docs_dir) / "eval-results.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(evaluation.results_markdown(results, with_llm), encoding="utf-8")
        except OSError as exc:
            raise CommandError(f"Sonuç dosyası yazılamadı ({out}): {exc}")
        self.stdout.write(f"Kaydedildi: {out}")
        if evaluation.update_readme(options["readme"], evaluation.readme_block(results, with_llm)):
            self.stdout.write(f"README güncellendi: {options['readme']}")
        else:
            self.stdout.write(self.style.WARNING(
                f"README güncellenmedi ({options['readme']} yok ya da eval işaretleri eksik)."))
