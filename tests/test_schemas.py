import pytest
from pydantic import ValidationError

from schemas import ChatMessage, ModelConfig, ModelResponse

from .conftest import FAKE_KEY, make_config


def test_chat_message_valid():
    msg = ChatMessage(role="user", content="hola")
    assert msg.role == "user"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": "tool", "content": "x"},
        {"role": "user", "content": ""},
        {"role": "user"},
    ],
)
def test_chat_message_invalid(kwargs):
    with pytest.raises(ValidationError):
        ChatMessage(**kwargs)


def test_model_response_defaults():
    r = ModelResponse(content="ok", provider="openai", model="m")
    assert r.usage.input_tokens is None and r.finish_reason is None


@pytest.mark.parametrize("temperature", [0, 1.0, 2])
def test_temperature_valid_for_openai(temperature):
    assert make_config("openai", temperature=temperature).temperature == temperature


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_out_of_range(temperature):
    with pytest.raises(ValidationError):
        make_config("openai", temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -5, 32769])
def test_max_tokens_invalid(max_tokens):
    with pytest.raises(ValidationError):
        make_config(max_tokens=max_tokens)


@pytest.mark.parametrize("field,value", [("timeout_s", 0), ("max_retries", -1), ("max_retries", 11)])
def test_other_limits_invalid(field, value):
    with pytest.raises(ValidationError):
        make_config(**{field: value})


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        make_config(provider="gemini")


def test_anthropic_rejects_temperature():
    with pytest.raises(ValidationError, match="anthropic"):
        make_config("anthropic", temperature=0.5)


def test_api_key_is_secret():
    config = make_config()
    assert FAKE_KEY not in repr(config)
    assert FAKE_KEY not in str(config)
    assert FAKE_KEY not in config.model_dump_json()
    assert config.api_key.get_secret_value() == FAKE_KEY


def test_empty_api_key_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="m", api_key="")
