import pytest

from clients.anthropic_client import AnthropicClient
from clients.base import BaseLLMClient
from clients.openai_client import OpenAIClient
from config import load_config
from errors import ConfigurationError
from llm_manager import AsyncLLMManager, create_client

from .conftest import FAKE_KEY, make_config


@pytest.mark.parametrize("provider,cls", [("openai", OpenAIClient), ("anthropic", AnthropicClient)])
def test_factory_selects_provider(provider, cls):
    client = create_client(make_config(provider))
    assert type(client) is cls
    assert isinstance(client, BaseLLMClient)


def test_base_client_is_abstract():
    with pytest.raises(TypeError):
        BaseLLMClient(make_config())  # type: ignore[abstract]


def test_manager_from_env_selects_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    manager = AsyncLLMManager.from_env()
    assert manager.config.provider == "anthropic"


def test_cli_override_switches_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    assert load_config(provider="openai").provider == "openai"


def test_missing_api_key_is_controlled_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_config()


def test_missing_model_and_invalid_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    with pytest.raises(ConfigurationError, match="LLM_MODEL"):
        load_config()
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        load_config(provider="gemini", model="m")


def test_optional_env_values_are_parsed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_MAX_TOKENS", "256")
    config = load_config()
    assert config.temperature == 0.7 and config.max_tokens == 256


def test_invalid_config_error_does_not_leak_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_TEMPERATURE", "5")
    with pytest.raises(ConfigurationError) as info:
        load_config()
    assert "temperature" in str(info.value)
    assert FAKE_KEY not in str(info.value)
    assert info.value.__cause__ is None
