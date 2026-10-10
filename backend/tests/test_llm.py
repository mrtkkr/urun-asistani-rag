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
                SimpleNamespace(values=[float(len(c.parts[0].text)), 4.0, 0.0])
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
    assert [c.parts[0].text for c in contents] == texts


def test_gemini_embed_raises_on_vector_count_mismatch(settings):
    from types import SimpleNamespace

    settings.EMBED_DIM = 3
    provider = _gemini_with_fake_client(settings)
    provider.client.embed_content = lambda **kw: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0, 0.0, 0.0])]
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        provider.embed(["a", "b", "c"])
