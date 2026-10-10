import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

# Outside Docker a .env at the repo root is picked up; inside Docker the
# variables arrive through compose's env_file and this is a no-op.
load_dotenv(REPO_ROOT / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(int(default))).lower() in ("1", "true", "yes")


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.staticfiles",
    "rest_framework",
    "catalog",
    "rag",
    "llm",
    "chat",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "urun_asistani"),
        "USER": os.environ.get("POSTGRES_USER", "urun"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "db"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LANGUAGE_CODE = "tr"
TIME_ZONE = "Europe/Istanbul"
USE_TZ = True
STATIC_URL = "static/"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
}

# --- Data files ---
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))

# --- LLM ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Ingest (free-tier friendly defaults) ---
INGEST_BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "20"))
INGEST_BATCH_PAUSE = float(os.environ.get("INGEST_BATCH_PAUSE", "1"))
INGEST_MAX_RETRIES = int(os.environ.get("INGEST_MAX_RETRIES", "5"))
INGEST_RETRY_WAIT = float(os.environ.get("INGEST_RETRY_WAIT", "15"))
