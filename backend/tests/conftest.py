import pytest


@pytest.fixture(autouse=True)
def _fake_llm(settings):
    """Guarantee no test can reach a real provider."""
    settings.LLM_PROVIDER = "fake"
    settings.GEMINI_API_KEY = ""


@pytest.fixture
def catalog(db):
    from django.core.management import call_command

    call_command("load_catalog")
