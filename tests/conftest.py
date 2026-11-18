from types import SimpleNamespace

import httpx2 as httpx
import pytest

from schemas import ChatMessage, ModelConfig

FAKE_KEY = "test-key-not-real"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los tests nunca leen .env ni variables reales del entorno."""
    for var in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "LLM_TIMEOUT_S",
        "LLM_MAX_RETRIES",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("config.load_dotenv", lambda *a, **k: False)


@pytest.fixture
def messages() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="¿Qué es la entropía?")]


def make_config(provider: str = "openai", **overrides) -> ModelConfig:
    values = {
        "provider": provider,
        "model": "test-model",
        "api_key": FAKE_KEY,
        "retry_base_delay_s": 0,
        "max_retries": 2,
    }
    values.update(overrides)
    return ModelConfig(**values)


def http_response(status: int, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.example.test/v1/chat")
    return httpx.Response(status, headers=headers or {}, request=request)


def http_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.example.test/v1/chat")


async def agen(*items, error: Exception | None = None):
    """Async generator de prueba: emite `items` y opcionalmente falla al final."""
    for item in items:
        yield item
    if error is not None:
        raise error


async def collect(aiter) -> list[str]:
    return [chunk async for chunk in aiter]


ns = SimpleNamespace
