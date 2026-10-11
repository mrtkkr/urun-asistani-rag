"""Measure retrieval and tool use over the questions in data/eval.json.

Search-only runs never call the chat model. The LLM part (tool calls, citations) is opt-in
because it spends free-tier quota.
"""
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime

from django.conf import settings

from llm import LLMError
from rag.search import search

from chat.prompts import group_by_source
from chat.service import ChatUnavailableError, answer_question

TYPES = ("katalog", "sss", "fiyat", "stok", "katalog_disi")
TOOL_FOR_TYPE = {"fiyat": "get_price", "stok": "check_stock"}
RANK_DEPTH = 20  # chunks fetched per question; enough to cover the top 4 distinct sources
README_START, README_END = "<!-- eval:start -->", "<!-- eval:end -->"
CASE_HEADER = ["id", "tür", "en iyi skor / kaynak", "hit@1", "hit@4", "eşik", "araç", "atıf"]
NOT_MEASURED = "ölçülemedi"


@dataclass
class CaseResult:
    case: dict
    ranked: list[str]            # distinct sources, best first, as "type:id"
    top_score: float | None
    passed: bool                 # at least one chunk reaches SEARCH_MIN_SCORE
    search_ok: bool = True
    llm: str = "skipped"         # skipped | ok | unmeasured
    tool_ok: bool | None = None
    cited: bool | None = None

    @property
    def expected(self) -> list[str]:
        return self.case["expected_sources"]

    @property
    def answerable(self) -> bool:
        return self.case["type"] != "katalog_disi"


def load_cases(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        if c["type"] not in TYPES:
            raise ValueError(f"{c['id']}: unknown type {c['type']!r}")
    return cases


def _ask_model(case, passed, provider, result: CaseResult) -> None:
    try:
        out = answer_question(case["question"], [], provider=provider, results=passed)
    except (LLMError, ChatUnavailableError):
        result.llm = "unmeasured"
        return
    result.llm = "ok"
    result.cited = any(s["cited"] for s in out["sources"])
    tool = TOOL_FOR_TYPE.get(case["type"])
    if tool:
        result.tool_ok = any(
            t["name"] == tool and t["args"].get("product_slug") == case["expected_slug"]
            for t in out["tool_calls"]
        )


def run(cases, provider, *, with_llm=False, pause=0.0, log=lambda msg: None) -> list[CaseResult]:
    results, asked = [], False
    for case in cases:
        try:
            ranked = search(case["question"], k=RANK_DEPTH, min_score=-1.0, provider=provider)
        except LLMError:
            results.append(CaseResult(case, [], None, False, search_ok=False,
                                      llm="unmeasured" if with_llm and case["type"] != "katalog_disi" else "skipped"))
            log(f"{case['id']}: arama {NOT_MEASURED}")
            continue
        firsts = [group[0].chunk for group in group_by_source(ranked, lambda r: r.chunk)]
        passed = [r for r in ranked[:settings.SEARCH_TOP_K] if r.score >= settings.SEARCH_MIN_SCORE]
        result = CaseResult(case, [f"{c.source_type}:{c.source_id}" for c in firsts],
                            ranked[0].score if ranked else None, bool(passed))
        if with_llm and result.answerable:
            if asked:
                time.sleep(pause)  # stay under the free-tier request limit
            asked = True
            _ask_model(case, passed, provider, result)
        results.append(result)
        log(f"{case['id']}: tamam")
    return results


def _hit(r: CaseResult, k: int) -> bool:
    return any(s in r.expected for s in r.ranked[:k])


def _row(name, measured, hit, pool) -> dict:
    """measured: the cases that could be scored; pool: all cases the metric covers."""
    return {"name": name, "hit": sum(1 for r in measured if hit(r)), "total": len(measured),
            "unmeasured": len(pool) - len(measured)}


def summarize(results: list[CaseResult], with_llm: bool) -> list[dict]:
    """Metric rows {"name", "hit", "total", "unmeasured"}; total is None when not measured."""
    answerable = [r for r in results if r.answerable]
    off_topic = [r for r in results if not r.answerable]
    searched = lambda pool: [r for r in pool if r.search_ok]
    rows = [
        _row("hit@1 (ilk kaynak doğru)", searched(answerable), lambda r: _hit(r, 1), answerable),
        _row("hit@4 (ilk 4 kaynakta doğru)", searched(answerable), lambda r: _hit(r, 4), answerable),
        _row("Alakalı soruda eşiği geçme", searched(answerable), lambda r: r.passed, answerable),
        _row("Katalog dışı soruda eşikte elenme", searched(off_topic), lambda r: not r.passed, off_topic),
    ]
    if with_llm:
        tools = [r for r in results if r.case["type"] in TOOL_FOR_TYPE]
        rows.append(_row("Doğru araç + doğru slug", [r for r in tools if r.llm == "ok"],
                         lambda r: r.tool_ok, tools))
        rows.append(_row("Yanıtta atıf bulunma", [r for r in answerable if r.llm == "ok"],
                         lambda r: r.cited, answerable))
    else:
        rows += [{"name": n, "hit": 0, "total": None, "unmeasured": 0}
                 for n in ("Doğru araç + doğru slug", "Yanıtta atıf bulunma")]
    return rows


def _ratio(row) -> str:
    if row["total"] is None:
        return "— (--with-llm ile ölçülür)"
    if row["total"] == 0:
        return NOT_MEASURED
    text = f"{row['hit']}/{row['total']} (%{round(100 * row['hit'] / row['total'])})"
    return text + (f", {row['unmeasured']} soru {NOT_MEASURED}" if row["unmeasured"] else "")


def _mark(value) -> str:
    return "—" if value is None else ("✓" if value else "✗")


def _case_cells(r: CaseResult, with_llm: bool) -> list[str]:
    if not r.search_ok:
        return [r.case["id"], r.case["type"], NOT_MEASURED, "", "", "", "", ""]
    top = r.ranked[0] if r.ranked else "—"
    score = f"{r.top_score:.2f}" if r.top_score is not None else "—"
    if r.answerable:
        search_cells = [_mark(_hit(r, 1)), _mark(_hit(r, 4)), _mark(r.passed)]
    else:
        search_cells = ["—", "—", "GEÇTİ" if r.passed else "elendi"]
    llm_cells = ["", ""]
    if with_llm and r.answerable:
        llm_cells = [NOT_MEASURED] * 2 if r.llm != "ok" else [_mark(r.tool_ok), _mark(r.cited)]
    return [r.case["id"], r.case["type"], f"{score} {top}", *search_cells, *llm_cells]


def _text_table(header, rows) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    def fmt(row):
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip()

    return "\n".join([fmt(header), fmt(["-" * w for w in widths]), *map(fmt, rows)])


def _md_table(header, rows) -> str:
    def esc(c):
        return str(c).replace("|", "\\|")

    return "\n".join([
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
        *("| " + " | ".join(esc(c) for c in row) + " |" for row in rows),
    ])


def _summary_md(results, with_llm) -> str:
    return _md_table(["Ölçüt", "Sonuç"], [[m["name"], _ratio(m)] for m in summarize(results, with_llm)])


def terminal_report(results, with_llm) -> str:
    cases = _text_table(CASE_HEADER, [_case_cells(r, with_llm) for r in results])
    metrics = _text_table(["Ölçüt", "Sonuç"], [[m["name"], _ratio(m)] for m in summarize(results, with_llm)])
    return f"{cases}\n\n{metrics}"


def _model_lines(with_llm: bool) -> list[str]:
    embed = f"- Embedding modeli: `{settings.EMBED_MODEL}` ({settings.EMBED_DIM} boyut)"
    if not with_llm:
        return ["- Sohbet modeli: çağrılmadı (yalnızca arama ölçüldü)", embed]
    return [f"- Sohbet modeli: `{settings.CHAT_MODEL}` (yedek: `{settings.CHAT_MODEL_FALLBACK or 'yok'}`, "
            f"düşünme seviyesi: `{settings.CHAT_THINKING_LEVEL or 'model varsayılanı'}`)", embed]


def results_markdown(results, with_llm, now=None) -> str:
    now = now or datetime.now()
    detail = [[r.case["id"], r.case["type"], r.case["question"], *_case_cells(r, with_llm)[2:]] for r in results]
    return "\n".join([
        "# Değerlendirme sonuçları", "",
        f"- Tarih: {now:%Y-%m-%d %H:%M}",
        f"- Mod: {'arama + LLM (--with-llm)' if with_llm else 'yalnızca arama'}",
        *_model_lines(with_llm),
        f"- Arama eşiği (SEARCH_MIN_SCORE): {settings.SEARCH_MIN_SCORE}, top-k: {settings.SEARCH_TOP_K}",
        f"- Soru sayısı: {len(results)}", "",
        "## Özet", "", _summary_md(results, with_llm), "",
        "## Soru bazında", "",
        _md_table(CASE_HEADER[:2] + ["soru"] + CASE_HEADER[2:], detail), "",
        "hit@k: beklenen kaynak, benzerliğe göre sıralı ilk k *farklı* kaynak içinde. "
        "eşik: en az bir parça eşiği geçti (katalog dışı sorularda 'elendi' beklenen sonuçtur). "
        f"{NOT_MEASURED}: sağlayıcı hatası; ilgili oran hesabına katılmaz.", "",
    ])


def readme_block(results, with_llm, now=None) -> str:
    now = now or datetime.now()
    mode = "arama + LLM" if with_llm else "yalnızca arama"
    chat = f"`{settings.CHAT_MODEL}`" if with_llm else "çağrılmadı"
    return (f"{README_START}\n"
            f"Son ölçüm: {now:%Y-%m-%d} ({mode}) · sohbet modeli: {chat} · embedding: "
            f"`{settings.EMBED_MODEL}` · eşik: {settings.SEARCH_MIN_SCORE} · {len(results)} soru\n\n"
            f"{_summary_md(results, with_llm)}\n\n"
            f"Soru bazında ayrıntı: [docs/eval-results.md](docs/eval-results.md)\n"
            f"{README_END}")


def update_readme(path, block: str) -> bool:
    """Replace the text between the eval markers; False when the file or the markers are missing."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
    except FileNotFoundError:
        return False
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        return False
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(pattern.sub(lambda _: block, text, count=1))
    return True
