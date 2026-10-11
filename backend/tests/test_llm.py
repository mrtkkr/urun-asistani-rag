import pytest
from django.conf import settings

from llm import get_provider
from llm.base import ChatResponse
from llm.fake import FakeProvider


def test_factory_returns_fake_provider():
    assert isinstance(get_provider(), FakeProvider)


def test_factory_rejects_unknown_provider(settings):
    settings.LLM_PROVIDER = "nope"
    with pytest.raises(ValueError):
        get_provider()


def test_gemini_requires_api_key(settings):
    settings.LLM_PROVIDER = "gemini"
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_provider()


def test_fake_embed_is_deterministic_and_sized():
    a, b = FakeProvider().embed(["merhaba", "merhaba"])
    assert a == b and len(a) == settings.EMBED_DIM


def test_fake_chat_returns_scripted_then_echo():
    provider = FakeProvider(responses=[ChatResponse(text="scripted")])
    messages = [{"role": "user", "content": "selam"}]
    assert provider.chat(messages).text == "scripted"
    assert provider.chat(messages).text == "fake: selam"


def _strip_format(text):
    """Drop the document prefix ("title: none | text: ") so tests see the raw text."""
    return text.split(" | text: ", 1)[-1]


class _FakeGeminiClient:
    """Stands in for genai.Client; records embed_content kwargs, returns raw 3-d vectors."""

    def __init__(self):
        self.calls = []
        self.models = self

    def embed_content(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(len(_strip_format(c.parts[0].text))), 4.0, 0.0])
                for c in kwargs["contents"]
            ]
        )


def _gemini_with_fake_client(settings):
    from llm.gemini import GeminiProvider

    settings.GEMINI_API_KEY = "test-key-not-real"
    provider = GeminiProvider()
    provider.client = _FakeGeminiClient()
    return provider


def test_gemini_embed_sends_output_dimensionality(settings):
    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    provider.embed(["a", "b"])
    config = provider.client.calls[0]["config"]
    assert config.output_dimensionality == 3
    assert provider.client.calls[0]["model"] == settings.EMBED_MODEL


def test_gemini_embed_l2_normalizes(settings):
    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    vectors = provider.embed(["abc"])
    assert vectors[0] == pytest.approx([0.6, 0.8, 0.0])


def test_gemini_embed_returns_one_vector_per_text_in_order(settings):
    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    texts = ["a", "bb", "ccc", "dddd"]
    vectors = provider.embed(texts)
    assert len(vectors) == len(texts)
    # The fake encodes text length in the first component, so order is observable.
    firsts = [v[0] / v[1] for v in vectors]
    assert firsts == sorted(firsts) and len(set(firsts)) == 4
    # Each text must be its own Content, not a bare list of strings.
    contents = provider.client.calls[0]["contents"]
    assert [_strip_format(c.parts[0].text) for c in contents] == texts


def test_gemini_embed_raises_on_vector_count_mismatch(settings):
    from types import SimpleNamespace

    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    provider.client.embed_content = lambda **kw: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0, 0.0, 0.0])]
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        provider.embed(["a", "b", "c"])


def test_gemini_document_embedding_uses_title_text_format(settings):
    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    provider.embed(["Süt ürünü."], task="document", titles=["Beyaz Peynir"])
    sent = [c.parts[0].text for c in provider.client.calls[0]["contents"]]
    assert sent == ["title: Beyaz Peynir | text: Süt ürünü."]


def test_gemini_document_without_title_uses_none(settings):
    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    provider.embed(["metin"])
    assert provider.client.calls[0]["contents"][0].parts[0].text == "title: none | text: metin"


def test_gemini_query_embedding_uses_search_prefix_and_no_task_type(settings):
    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    provider.embed(["kaşar nasıl saklanır?"], task="query")
    call = provider.client.calls[0]
    assert call["contents"][0].parts[0].text == "task: search result | query: kaşar nasıl saklanır?"
    assert call["config"].task_type is None  # gemini-embedding-2 rejects task_type


def test_embed_rejects_unknown_task(settings):
    settings.EMBED_DIM = 3
    with pytest.raises(ValueError):
        _gemini_with_fake_client(settings).embed(["a"], task="nope")
    with pytest.raises(ValueError):
        FakeProvider().embed(["a"], task="nope")


class _FakeGeminiChatClient:
    def __init__(self, parts):
        from types import SimpleNamespace

        self.calls = []
        self.models = self
        self._response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_gemini_chat_sends_system_instruction_and_history(settings):
    from types import SimpleNamespace

    provider = _gemini_with_fake_client(settings)
    provider.client = _FakeGeminiChatClient([SimpleNamespace(function_call=None, text="Merhaba!")])
    response = provider.chat([
        {"role": "system", "content": "Türkçe konuş."},
        {"role": "user", "content": "selam"},
        {"role": "assistant", "content": "merhaba"},
        {"role": "user", "content": "kaşar?"},
    ])
    call = provider.client.calls[0]
    assert response.text == "Merhaba!"
    assert call["config"].system_instruction == "Türkçe konuş."
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][2].parts[0].text == "kaşar?"
    assert call["model"] == settings.CHAT_MODEL


def test_fake_chat_records_calls_for_inspection():
    provider = FakeProvider()
    messages = [{"role": "user", "content": "x"}]
    provider.chat(messages, tools=[{"name": "t"}])
    assert provider.last_chat == {"messages": messages, "tools": [{"name": "t"}]}


def test_gemini_chat_returns_tool_calls_and_keeps_raw_content(settings):
    from types import SimpleNamespace

    provider = _gemini_with_fake_client(settings)
    fc = SimpleNamespace(name="get_price", args={"product_slug": "kasar-eski"}, id=None)
    parts = [SimpleNamespace(function_call=fc, text=None, thought=None)]
    provider.client = _FakeGeminiChatClient(parts)
    tools = [{"name": "get_price", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    response = provider.chat([{"role": "user", "content": "fiyat?"}], tools)
    assert [(c.name, c.args) for c in response.tool_calls] == [("get_price", {"product_slug": "kasar-eski"})]
    assert response.raw.parts == parts  # replayed verbatim (thought signatures)
    config = provider.client.calls[0]["config"]
    assert config.tools[0].function_declarations[0].name == "get_price"
    assert config.automatic_function_calling.disable is True


def test_gemini_replays_raw_model_content_and_groups_tool_results():
    from google.genai import types

    from llm.gemini import GeminiProvider

    raw = types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="get_price", args={"product_slug": "a"})),
        types.Part(function_call=types.FunctionCall(name="check_stock", args={"product_slug": "a"})),
    ])
    contents = GeminiProvider._contents([
        {"role": "user", "content": "soru"},
        {"role": "assistant", "content": "", "tool_calls": [], "raw": raw},
        {"role": "tool", "name": "get_price", "tool_call_id": "", "content": '{"price": "1"}'},
        {"role": "tool", "name": "check_stock", "tool_call_id": "", "content": '{"stock": 2}'},
    ])
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1] is raw
    assert [p.function_response.name for p in contents[2].parts] == ["get_price", "check_stock"]


@pytest.mark.parametrize("code, expected", [(503, "ProviderUnavailableError"), (429, "RateLimitError")])
def test_gemini_chat_maps_http_errors(settings, code, expected):
    from google.genai import errors

    import llm.base as base

    provider = _gemini_with_fake_client(settings)

    def boom(**kwargs):
        raise errors.APIError(code, {"error": {"message": "x", "status": "UNAVAILABLE"}})

    provider.client = type("C", (), {"models": type("M", (), {"generate_content": staticmethod(boom)})})()
    with pytest.raises(getattr(base, expected)):
        provider.chat([{"role": "user", "content": "x"}])


def test_gemini_chat_maps_timeouts_to_unavailable(settings):
    import httpx

    from llm.base import ProviderUnavailableError

    provider = _gemini_with_fake_client(settings)

    def boom(**kwargs):
        raise httpx.ReadTimeout("slow")

    provider.client = type("C", (), {"models": type("M", (), {"generate_content": staticmethod(boom)})})()
    with pytest.raises(ProviderUnavailableError):
        provider.chat([{"role": "user", "content": "x"}])


def test_fake_provider_raises_scripted_exceptions():
    from llm.base import ProviderUnavailableError

    provider = FakeProvider(responses=[ProviderUnavailableError("503"), ChatResponse(text="ok")])
    with pytest.raises(ProviderUnavailableError):
        provider.chat([{"role": "user", "content": "x"}])
    assert provider.chat([{"role": "user", "content": "x"}]).text == "ok"
