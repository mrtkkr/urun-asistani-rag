import json

import pytest
from rest_framework.test import APIClient

from catalog.models import Product
from chat.service import answer_question, search_query
from chat.tools import TOOL_DEFINITIONS, check_stock, get_price, run_tool
from llm.base import ChatResponse, ToolCall
from llm.fake import FakeProvider
from rag.models import Chunk

URL = "/api/chat/"


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr("chat.service.get_provider", lambda: fake)
    monkeypatch.setattr("rag.search.get_provider", lambda: fake)
    return fake


@pytest.fixture
def kasar(catalog, provider):
    """A product chunk for Eski Kaşar that matches the query "eski kasar" exactly."""
    Chunk.objects.create(source_type="product", source_id="kasar-eski", title="Eski Kaşar",
                         text="Eski Kaşar bilgisi.", embedding=FakeProvider._vector("eski kasar fiyatı?"))
    return Product.objects.get(slug="kasar-eski")


def call(name, slug="kasar-eski", id="c1"):
    return ToolCall(name=name, args={"product_slug": slug}, id=id)


# --- tools ------------------------------------------------------------------

def test_tool_definitions_describe_both_tools():
    assert [t["name"] for t in TOOL_DEFINITIONS] == ["get_price", "check_stock"]
    assert all(t["parameters"]["required"] == ["product_slug"] for t in TOOL_DEFINITIONS)


def test_get_price_reads_the_database(kasar):
    kasar.price = "123.50"
    kasar.save()
    assert get_price("kasar-eski") == {
        "slug": "kasar-eski", "name": "Eski Kaşar", "price": "123.50", "currency": "TRY"}


def test_check_stock_reads_the_database(kasar):
    kasar.stock = 7
    kasar.save()
    assert check_stock("kasar-eski")["stock"] == 7
    assert check_stock("kasar-eski")["in_stock"] is True
    kasar.stock = 0
    kasar.save()
    assert check_stock("kasar-eski")["in_stock"] is False


def test_unknown_slug_returns_candidates_not_an_exception(catalog):
    result = get_price("eski-kasar")
    assert "error" in result and "price" not in result
    candidates = [c["slug"] for c in result["candidates"]]
    assert candidates[0] == "kasar-eski" and len(candidates) == 3


@pytest.mark.parametrize("bad", [None, "", 42])
def test_bad_slug_values_do_not_crash(catalog, bad):
    assert "candidates" in check_stock(bad)


def test_run_tool_handles_unknown_tool_and_bad_arguments(catalog):
    assert "error" in run_tool("drop_tables", {})
    assert "error" in run_tool("get_price", {"wrong": "x"})


# --- loop -------------------------------------------------------------------

def test_loop_runs_tool_and_feeds_result_back_to_model(kasar, provider):
    provider.responses = [
        ChatResponse(tool_calls=[call("get_price")], raw="RAW"),
        ChatResponse(text="Eski Kaşar fiyatı söylendi. [1]"),
    ]
    out = answer_question("eski kasar fiyatı?", [])
    assert out["answer"] == "Eski Kaşar fiyatı söylendi. [1]"
    assert out["tool_calls"] == [
        {"name": "get_price", "args": {"product_slug": "kasar-eski"},
         "result": get_price("kasar-eski")}]
    first, second = provider.chat_calls
    assert [t["name"] for t in first["tools"]] == ["get_price", "check_stock"]
    # The second request carries the assistant's tool request and the tool result.
    assert [m["role"] for m in second["messages"]][-2:] == ["assistant", "tool"]
    assert second["messages"][-2]["raw"] == "RAW"
    tool_msg = second["messages"][-1]
    assert tool_msg["name"] == "get_price" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"]) == get_price("kasar-eski")


def test_loop_handles_parallel_calls_in_order(kasar, provider):
    provider.responses = [
        ChatResponse(tool_calls=[call("get_price", id="a"), call("check_stock", id="b")]),
        ChatResponse(text="Tamam."),
    ]
    out = answer_question("eski kasar fiyatı?", [])
    assert [t["name"] for t in out["tool_calls"]] == ["get_price", "check_stock"]
    tool_ids = [m["tool_call_id"] for m in provider.chat_calls[1]["messages"] if m["role"] == "tool"]
    assert tool_ids == ["a", "b"]


def test_loop_stops_after_three_turns_with_polite_error(kasar, provider):
    provider.responses = [ChatResponse(tool_calls=[call("get_price", id=str(i))]) for i in range(10)]
    out = answer_question("eski kasar fiyatı?", [])
    assert len(provider.chat_calls) == 3
    assert "tamamlayamadım" in out["answer"]
    assert len(out["tool_calls"]) == 3 and out["sources"]


def test_unknown_slug_result_reaches_the_model(kasar, provider):
    provider.responses = [
        ChatResponse(tool_calls=[call("get_price", slug="eski-kasar")]),
        ChatResponse(tool_calls=[call("get_price", id="c2")]),
        ChatResponse(text="Bitti."),
    ]
    out = answer_question("eski kasar fiyatı?", [])
    assert "candidates" in out["tool_calls"][0]["result"]
    assert "price" in out["tool_calls"][1]["result"]
    # The fake keeps a reference to the growing list, so pick the first tool message.
    first_tool = next(m for m in provider.chat_calls[1]["messages"] if m["role"] == "tool")
    assert "candidates" in json.loads(first_tool["content"])


# --- follow-ups & citations ----------------------------------------------------

def test_search_query_joins_last_two_user_messages():
    history = [
        {"role": "user", "content": "ilk soru"},
        {"role": "assistant", "content": "cevap"},
        {"role": "user", "content": "eski kasar nasıl saklanır"},
        {"role": "assistant", "content": "soğukta"},
    ]
    assert search_query("peki fiyatı?", history) == "eski kasar nasıl saklanır\npeki fiyatı?"
    assert search_query("soru", []) == "soru"
    assert search_query("soru", [{"role": "assistant", "content": "x"}]) == "soru"


def test_follow_up_embeds_the_combined_query(kasar, provider):
    history = [{"role": "user", "content": "eski kasar"}, {"role": "assistant", "content": "x"}]
    answer_question("peki fiyatı?", history)
    assert provider.embed_calls == [["eski kasar\npeki fiyatı?"]]


def test_follow_up_via_api_finds_previous_product(kasar, provider, settings):
    settings.SEARCH_MIN_SCORE = 0.99
    Chunk.objects.filter(source_id="kasar-eski").update(
        embedding=FakeProvider._vector("eski kasar\npeki fiyatı?"))
    client = APIClient()
    bare = client.post(URL, {"message": "peki fiyatı?"}, format="json").json()
    assert bare["sources"] == []  # no history: nothing relevant
    history = [{"role": "user", "content": "eski kasar"}]
    full = client.post(URL, {"message": "peki fiyatı?", "history": history}, format="json").json()
    assert [s["id"] for s in full["sources"]] == ["kasar-eski"]


def test_cited_flag_follows_bracket_numbers(catalog, provider):
    vec = FakeProvider._vector("eski kasar")
    Chunk.objects.create(source_type="product", source_id="kasar-eski", title="Eski Kaşar",
                         text="a", chunk_index=0, embedding=vec)
    Chunk.objects.create(source_type="product", source_id="kasar-eski", title="Eski Kaşar",
                         text="b", chunk_index=1, embedding=vec)
    Chunk.objects.create(source_type="faq", source_id="kargo", title="Kargo", text="c", embedding=vec)
    provider.responses = [ChatResponse(text="Şöyle saklanır [1], [2].")]
    sources = answer_question("eski kasar", [])["sources"]
    assert len(sources) == 2  # two chunks of one product collapse into one source
    assert [s["cited"] for s in sources if s["id"] == "kasar-eski"] == [True]


def test_cited_is_false_when_not_referenced(kasar, provider):
    provider.responses = [ChatResponse(text="Atıfsız cevap.")]
    assert answer_question("eski kasar fiyatı?", [])["sources"][0]["cited"] is False
    provider.responses = [ChatResponse(text="Atıflı [1].")]
    assert answer_question("eski kasar fiyatı?", [])["sources"][0]["cited"] is True
