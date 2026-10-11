import json
from collections import Counter

import pytest
from django.conf import settings
from django.core.management import call_command

from chat import evaluation
from llm.base import ChatResponse, ProviderUnavailableError, RateLimitError, ToolCall
from llm.fake import FakeProvider
from rag.models import Chunk


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr("chat.service.get_provider", lambda: fake)
    monkeypatch.setattr("rag.search.get_provider", lambda: fake)
    monkeypatch.setattr("chat.management.commands.evaluate.get_provider", lambda: fake)
    return fake


@pytest.fixture
def eval_env(db, provider, tmp_path, catalog):
    """A tiny eval set whose questions embed exactly onto their expected chunks."""
    cases = [
        {"id": "k1", "type": "katalog", "question": "tulum peyniri nedir", "expected_sources": ["product:tulum-peyniri"]},
        {"id": "s1", "type": "sss", "question": "kargo kac gun", "expected_sources": ["faq:kargo"]},
        {"id": "f1", "type": "fiyat", "question": "kasar fiyati", "expected_sources": ["product:kasar-eski"],
         "expected_tool": "get_price", "expected_slug": "kasar-eski"},
        {"id": "d1", "type": "katalog_disi", "question": "hava nasil", "expected_sources": []},
    ]
    # k1 is answered by the wrong source on purpose: it must count as a miss.
    for source_type, source_id, text in [("product", "kasar-eski", "kasar fiyati"),
                                         ("faq", "kargo", "kargo kac gun"),
                                         ("product", "bal", "tulum peyniri nedir")]:
        Chunk.objects.create(source_type=source_type, source_id=source_id, title=source_id, text=text,
                             embedding=FakeProvider._vector(text))
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Başlık\n\nönce\n<!-- eval:start -->\nhenüz çalıştırılmadı\n<!-- eval:end -->\nsonra\n",
                      encoding="utf-8")
    return {"file": str(path), "docs": str(tmp_path / "docs"), "readme": readme, "tmp": tmp_path}


def run_command(env, *flags):
    call_command("evaluate", "--file", env["file"], "--docs-dir", env["docs"],
                 "--readme", str(env["readme"]), "--pause", "0", *flags)


def metric(results, name_start, with_llm=False):
    row = next(m for m in evaluation.summarize(results, with_llm) if m["name"].startswith(name_start))
    return row["hit"], row["total"], row["unmeasured"]


# --- the real data file -----------------------------------------------------------

def test_real_eval_file_is_consistent_with_catalog_and_faq():
    cases = evaluation.load_cases(settings.DATA_DIR / "eval.json")
    assert len(cases) == 18
    counts = Counter(c["type"] for c in cases)
    assert counts["katalog"] == 7 and counts["sss"] == 4 and counts["katalog_disi"] == 4
    assert counts["fiyat"] + counts["stok"] == 3
    assert len({c["id"] for c in cases}) == 18

    from rag.chunking import faq_chunks

    slugs = {p["slug"] for p in json.loads((settings.DATA_DIR / "products.json").read_text(encoding="utf-8"))}
    faq_ids = {s.source_id for s in faq_chunks((settings.DATA_DIR / "faq.md").read_text(encoding="utf-8"))}
    known = {f"product:{s}" for s in slugs} | {f"faq:{s}" for s in faq_ids}
    for c in cases:
        assert set(c["expected_sources"]) <= known, c["id"]
        if c["type"] == "katalog_disi":
            assert c["expected_sources"] == []
        else:
            assert c["expected_sources"], c["id"]
        if c["type"] in evaluation.TOOL_FOR_TYPE:
            assert c["expected_tool"] == evaluation.TOOL_FOR_TYPE[c["type"]]
            assert c["expected_slug"] in slugs


def test_unknown_case_type_is_rejected(tmp_path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps([{"id": "x", "type": "bilinmeyen", "question": "?", "expected_sources": []}]))
    with pytest.raises(ValueError):
        evaluation.load_cases(path)


# --- search-only mode --------------------------------------------------------------

def test_search_only_measures_hits_and_threshold_without_calling_the_chat_model(eval_env, provider, capsys):
    run_command(eval_env)
    assert provider.chat_calls == []
    out = capsys.readouterr().out
    assert "hit@1" in out and "k1" in out and "elendi" in out
    cases = evaluation.load_cases(eval_env["file"])
    results = evaluation.run(cases, provider)
    # k1's best source is the wrong one, s1 and f1 are right.
    assert metric(results, "hit@1") == (2, 3, 0)
    assert metric(results, "hit@4") == (2, 3, 0)
    assert metric(results, "Alakalı") == (3, 3, 0)  # all three questions pass the threshold
    assert metric(results, "Katalog dışı") == (1, 1, 0)  # the weather question is filtered out
    assert provider.chat_calls == []


def test_off_topic_question_that_passes_the_threshold_counts_as_a_miss(eval_env, provider):
    cases = [{"id": "d9", "type": "katalog_disi", "question": "kargo kac gun", "expected_sources": []}]
    assert metric(evaluation.run(cases, provider), "Katalog dışı") == (0, 1, 0)


def test_search_only_writes_report_and_updates_only_the_readme_block(eval_env):
    run_command(eval_env)
    report = (eval_env["tmp"] / "docs" / "eval-results.md").read_text(encoding="utf-8")
    assert "Değerlendirme sonuçları" in report and "yalnızca arama" in report
    assert settings.EMBED_MODEL in report and str(settings.SEARCH_MIN_SCORE) in report
    assert "tulum peyniri nedir" in report  # per-question rows
    readme = eval_env["readme"].read_text(encoding="utf-8")
    assert readme.startswith("# Başlık\n\nönce\n<!-- eval:start -->\n")
    assert readme.endswith("<!-- eval:end -->\nsonra\n")
    assert "hit@1" in readme and "henüz çalıştırılmadı" not in readme


def test_rerun_replaces_the_block_instead_of_appending(eval_env):
    run_command(eval_env)
    run_command(eval_env)
    assert eval_env["readme"].read_text(encoding="utf-8").count("<!-- eval:start -->") == 1


def test_missing_readme_markers_only_warn(eval_env, capsys):
    eval_env["readme"].write_text("işaret yok", encoding="utf-8")
    run_command(eval_env)
    assert eval_env["readme"].read_text(encoding="utf-8") == "işaret yok"
    assert "README güncellenmedi" in capsys.readouterr().out


def test_search_error_marks_question_unmeasured_and_keeps_rates_honest(eval_env, provider):
    cases = evaluation.load_cases(eval_env["file"])
    provider.fail_embed_times = 1  # the first question's embedding hits a 429
    results = evaluation.run(cases, provider)
    assert not results[0].search_ok
    assert metric(results, "hit@1") == (2, 2, 1)  # k1 left out, not counted as a miss


# --- --with-llm mode ---------------------------------------------------------------

def test_with_llm_scores_tool_call_with_slug_and_citation(eval_env, provider):
    provider.responses = [
        ChatResponse(text="Peynir [1].."),                                 # k1
        ChatResponse(text="Kargo 1-3 gün."),                               # s1, no citation
        ChatResponse(tool_calls=[ToolCall("get_price", {"product_slug": "kasar-eski"})]),  # f1
        ChatResponse(text="Fiyatı 329 TL [1]."),
    ]
    results = evaluation.run(evaluation.load_cases(eval_env["file"]), provider, with_llm=True)
    assert metric(results, "Doğru araç", True) == (1, 1, 0)
    assert metric(results, "Yanıtta atıf", True) == (2, 3, 0)
    assert len(provider.chat_calls) == 4  # the off-topic question never reaches the model


def test_wrong_tool_or_slug_is_a_miss(eval_env, provider):
    provider.responses = [
        ChatResponse(text="x"), ChatResponse(text="y"),
        ChatResponse(tool_calls=[ToolCall("check_stock", {"product_slug": "kasar-eski"})]),
        ChatResponse(text="z"),
    ]
    results = evaluation.run(evaluation.load_cases(eval_env["file"]), provider, with_llm=True)
    assert metric(results, "Doğru araç", True) == (0, 1, 0)


def test_provider_error_marks_question_unmeasured_without_distorting_rates(eval_env, provider, settings):
    settings.CHAT_MAX_RETRIES = 0
    provider.responses = [
        ChatResponse(text="Peynir [1]."),
        ProviderUnavailableError("503"),                                   # s1 cannot be measured
        ChatResponse(tool_calls=[ToolCall("get_price", {"product_slug": "kasar-eski"})]),
        ChatResponse(text="329 TL [1]."),
    ]
    results = evaluation.run(evaluation.load_cases(eval_env["file"]), provider, with_llm=True)
    assert results[1].llm == "unmeasured"
    assert metric(results, "Yanıtta atıf", True) == (2, 2, 1)
    assert metric(results, "Doğru araç", True) == (1, 1, 0)


def test_rate_limited_question_is_unmeasured_too(eval_env, provider, settings):
    settings.CHAT_MAX_RETRIES = 0
    provider.responses = [RateLimitError("429", status_code=429)]
    results = evaluation.run(evaluation.load_cases(eval_env["file"]), provider, with_llm=True)
    assert results[0].llm == "unmeasured"
    assert metric(results, "Yanıtta atıf", True)[2] == 1


def test_with_llm_command_pauses_between_model_calls(eval_env, provider, monkeypatch):
    sleeps = []
    monkeypatch.setattr("chat.evaluation.time.sleep", sleeps.append)
    call_command("evaluate", "--with-llm", "--pause", "7", "--file", eval_env["file"],
                 "--docs-dir", eval_env["docs"], "--readme", str(eval_env["readme"]))
    assert sleeps == [7.0, 7.0]  # three answerable questions, so two pauses
    readme = eval_env["readme"].read_text(encoding="utf-8")
    assert "Doğru araç" in readme and "(--with-llm ile ölçülür)" not in readme
    report = (eval_env["tmp"] / "docs" / "eval-results.md").read_text(encoding="utf-8")
    assert settings.CHAT_MODEL in report or "arama + LLM" in report
