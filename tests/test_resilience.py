import logging

import anthropic
import openai
import pytest

from clients.anthropic_client import AnthropicClient
from clients.openai_client import OpenAIClient
from errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    redact,
)

from .conftest import FAKE_KEY, agen, collect, http_request, http_response, make_config
from .test_providers import (
    FakeAnthropicStream,
    anthropic_message,
    fake_anthropic,
    fake_openai,
    openai_chunk,
    openai_completion,
)


def sdk_error(sdk, kind: str, status: int | None = None, headers: dict | None = None, body=None):
    if kind == "timeout":
        return sdk.APITimeoutError(request=http_request())
    if kind == "connection":
        return sdk.APIConnectionError(request=http_request())
    cls = {429: sdk.RateLimitError, 401: sdk.AuthenticationError, 400: sdk.BadRequestError}.get(
        status, sdk.APIStatusError
    )
    return cls("boom", response=http_response(status, headers), body=body)


# Cada proveedor se expone como (SDK, builder de client, success value, función para armar fallos).
def openai_setup(*effects):
    sdk = fake_openai(*effects)
    return OpenAIClient(make_config("openai"), client=sdk), sdk.chat.completions.create


def anthropic_setup(*effects):
    sdk = fake_anthropic(create=list(effects))
    return AnthropicClient(make_config("anthropic"), client=sdk), sdk.messages.create


PROVIDERS = [
    pytest.param(openai, openai_setup, openai_completion("ok"), id="openai"),
    pytest.param(anthropic, anthropic_setup, anthropic_message("ok"), id="anthropic"),
]


@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_retries_rate_limit_then_succeeds(sdk, setup, success, messages):
    client, create = setup(sdk_error(sdk, "status", 429, {"retry-after": "0"}), success)
    response = await client.generate(messages)
    assert response.content == "ok"
    assert create.await_count == 2


@pytest.mark.parametrize("kind", ["timeout", "connection"])
@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_retries_network_errors_then_succeeds(sdk, setup, success, kind, messages):
    client, create = setup(sdk_error(sdk, kind), success)
    assert (await client.generate(messages)).content == "ok"
    assert create.await_count == 2


@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_retries_server_errors(sdk, setup, success, messages):
    client, create = setup(sdk_error(sdk, "status", 503), success)
    assert (await client.generate(messages)).content == "ok"
    assert create.await_count == 2


@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_rate_limit_exhausts_retries_and_raises(sdk, setup, success, messages):
    errors = [sdk_error(sdk, "status", 429) for _ in range(3)]  # 1 intento + 2 reintentos
    client, create = setup(*errors)
    with pytest.raises(LLMRateLimitError):
        await client.generate(messages)
    assert create.await_count == 3


@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_timeout_exhausts_retries_and_raises(sdk, setup, success, messages):
    client, create = setup(*[sdk_error(sdk, "timeout") for _ in range(3)])
    with pytest.raises(LLMTimeoutError):
        await client.generate(messages)
    assert create.await_count == 3


@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_connection_error_exhausts_retries_and_raises(sdk, setup, success, messages):
    client, create = setup(*[sdk_error(sdk, "connection") for _ in range(3)])
    with pytest.raises(LLMConnectionError):
        await client.generate(messages)
    assert create.await_count == 3


@pytest.mark.parametrize("status,expected", [(401, LLMAuthenticationError), (400, LLMBadRequestError)])
@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_permanent_errors_are_not_retried(sdk, setup, success, status, expected, messages):
    client, create = setup(sdk_error(sdk, "status", status), success)
    with pytest.raises(expected):
        await client.generate(messages)
    assert create.await_count == 1


async def test_openai_insufficient_quota_is_permanent(messages):
    err = sdk_error(openai, "status", 429, body={"code": "insufficient_quota"})
    client, create = openai_setup(err, openai_completion("ok"))
    with pytest.raises(LLMBadRequestError, match="cuota"):
        await client.generate(messages)
    assert create.await_count == 1


async def test_retry_after_header_is_honored(monkeypatch, messages):
    waits: list[float] = []

    async def fake_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("clients.base.asyncio.sleep", fake_sleep)
    client, _ = openai_setup(sdk_error(openai, "status", 429, {"retry-after": "7"}), openai_completion("ok"))
    await client.generate(messages)
    assert waits == [7.0]


async def test_exponential_backoff_without_retry_after(monkeypatch, messages):
    waits: list[float] = []

    async def fake_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("clients.base.asyncio.sleep", fake_sleep)
    sdk = fake_openai(sdk_error(openai, "connection"), sdk_error(openai, "connection"), openai_completion("ok"))
    client = OpenAIClient(make_config("openai", retry_base_delay_s=1), client=sdk)
    await client.generate(messages)
    assert waits == [1, 2]


def test_sdk_internal_retries_are_disabled():
    """Evita retries duplicados: el único nivel de retry es el de BaseLLMClient."""
    assert OpenAIClient(make_config("openai"))._client.max_retries == 0
    assert AnthropicClient(make_config("anthropic"))._client.max_retries == 0


# ---------- Streaming ----------
async def test_openai_stream_retries_if_failure_before_first_chunk(messages):
    sdk = fake_openai(sdk_error(openai, "connection"), agen(openai_chunk("a"), openai_chunk("b")))
    client = OpenAIClient(make_config("openai"), client=sdk)
    assert await collect(client.stream(messages)) == ["a", "b"]


async def test_openai_stream_error_midway_is_not_retried(messages):
    failing = agen(openai_chunk("a"), error=sdk_error(openai, "connection"))
    sdk = fake_openai(failing, agen(openai_chunk("a")))
    client = OpenAIClient(make_config("openai"), client=sdk)
    received: list[str] = []
    with pytest.raises(LLMConnectionError):
        async for chunk in client.stream(messages):
            received.append(chunk)
    assert received == ["a"]
    assert sdk.chat.completions.create.await_count == 1


async def test_anthropic_stream_error_midway_is_not_retried(messages):
    failing = FakeAnthropicStream(agen("a", error=sdk_error(anthropic, "connection")))
    sdk = fake_anthropic(streams=[failing, FakeAnthropicStream(agen("a"))])
    client = AnthropicClient(make_config("anthropic"), client=sdk)
    received: list[str] = []
    with pytest.raises(LLMConnectionError):
        async for chunk in client.stream(messages):
            received.append(chunk)
    assert received == ["a"]


async def test_anthropic_stream_rate_limit_before_first_chunk_is_retried(messages):
    first = FakeAnthropicStream(agen(error=sdk_error(anthropic, "status", 429)))
    sdk = fake_anthropic(streams=[first, FakeAnthropicStream(agen("ok"))])
    client = AnthropicClient(make_config("anthropic"), client=sdk)
    assert await collect(client.stream(messages)) == ["ok"]


async def test_stream_auth_error_is_permanent(messages):
    sdk = fake_openai(sdk_error(openai, "status", 401), agen(openai_chunk("x")))
    client = OpenAIClient(make_config("openai"), client=sdk)
    with pytest.raises(LLMAuthenticationError):
        await collect(client.stream(messages))
    assert sdk.chat.completions.create.await_count == 1


# ---------- Secretos ----------
async def test_error_messages_do_not_contain_api_key(messages):
    client, _ = openai_setup(sdk_error(openai, "status", 401))
    with pytest.raises(LLMAuthenticationError) as info:
        await client.generate(messages)
    assert FAKE_KEY not in str(info.value)


# ---------- Saneamiento de mensajes ----------
# Valores armados en runtime para que el código fuente no contenga literales con forma de credencial.
FAKE_SK = "sk-" + "live" + "1234567890ABCD"
FAKE_BEARER = "tok" + ".en-" + "123"
LEAKY = f"{FAKE_KEY} {FAKE_SK} Authorization: Bearer {FAKE_BEARER}"


def leaky_error(sdk, kind: str):
    if kind == "connection":
        return sdk.APIConnectionError(message=f"fallo con {LEAKY}", request=http_request())
    if kind == "timeout":
        return sdk.APITimeoutError(request=http_request())
    cls = sdk.AuthenticationError if kind == "auth" else sdk.BadRequestError
    status = 401 if kind == "auth" else 400
    return cls(f"rechazado {LEAKY}", response=http_response(status), body=None)


@pytest.mark.parametrize("kind", ["connection", "auth", "bad_request"])
@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_error_messages_are_redacted(sdk, setup, success, kind, messages):
    client, _ = setup(*[leaky_error(sdk, kind) for _ in range(3)])
    with pytest.raises(LLMError) as info:
        await client.generate(messages)
    text = str(info.value)
    for leaked in (FAKE_KEY, FAKE_SK, FAKE_BEARER):
        assert leaked not in text
    assert "[REDACTED]" in text


@pytest.mark.parametrize("sdk,setup,success", PROVIDERS)
async def test_retry_logs_do_not_contain_secrets(sdk, setup, success, messages, caplog):
    client, _ = setup(leaky_error(sdk, "connection"), success)
    with caplog.at_level(logging.WARNING):
        await client.generate(messages)
    assert "Reintento 1/2" in caplog.text
    assert FAKE_KEY not in caplog.text and FAKE_SK not in caplog.text


def test_redact_helper():
    assert redact("a MYKEY b", "MYKEY") == "a [REDACTED] b"
    assert FAKE_SK not in redact(f"clave {FAKE_SK}")
    assert redact("sin secretos aquí") == "sin secretos aquí"
