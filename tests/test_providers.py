from unittest.mock import AsyncMock

import pytest

from clients.anthropic_client import AnthropicClient
from clients.openai_client import OpenAIClient
from errors import LLMInvalidResponseError

from .conftest import agen, collect, make_config, ns


# ---------- OpenAI ----------
def openai_completion(content="respuesta", choices=True):
    choice = ns(message=ns(content=content), finish_reason="stop")
    return ns(
        model="gpt-test",
        choices=[choice] if choices else [],
        usage=ns(prompt_tokens=7, completion_tokens=3),
    )


def openai_chunk(text):
    return ns(choices=[ns(delta=ns(content=text))])


def fake_openai(*side_effect):
    return ns(chat=ns(completions=ns(create=AsyncMock(side_effect=list(side_effect)))))


async def test_openai_generate_adapts_response(messages):
    sdk = fake_openai(openai_completion("hola mundo"))
    client = OpenAIClient(make_config("openai", temperature=0.2), client=sdk)

    response = await client.generate(messages)

    assert response.content == "hola mundo"
    assert (response.provider, response.model, response.finish_reason) == ("openai", "gpt-test", "stop")
    assert (response.usage.input_tokens, response.usage.output_tokens) == (7, 3)
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert kwargs["max_completion_tokens"] == 1024
    assert kwargs["temperature"] == 0.2


async def test_openai_omits_temperature_when_unset(messages):
    sdk = fake_openai(openai_completion())
    await OpenAIClient(make_config("openai"), client=sdk).generate(messages)
    assert "temperature" not in sdk.chat.completions.create.call_args.kwargs


async def test_openai_stream_yields_only_text_deltas(messages):
    stream = agen(openai_chunk("La "), ns(choices=[]), openai_chunk(None), openai_chunk("entropía"))
    sdk = fake_openai(stream)
    client = OpenAIClient(make_config("openai"), client=sdk)

    assert await collect(client.stream(messages)) == ["La ", "entropía"]
    assert sdk.chat.completions.create.call_args.kwargs["stream"] is True


@pytest.mark.parametrize("completion", [openai_completion(choices=False), openai_completion(content=None)])
async def test_openai_invalid_response(messages, completion):
    client = OpenAIClient(make_config("openai"), client=fake_openai(completion))
    with pytest.raises(LLMInvalidResponseError):
        await client.generate(messages)


# ---------- Anthropic ----------
def anthropic_message(*texts):
    blocks = [ns(type="text", text=t) for t in texts]
    return ns(
        model="claude-test",
        content=blocks,
        stop_reason="end_turn",
        usage=ns(input_tokens=5, output_tokens=9),
    )


class FakeAnthropicStream:
    def __init__(self, source):
        self._source = source

    async def __aenter__(self):
        return ns(text_stream=self._source)

    async def __aexit__(self, *exc):
        return False


def fake_anthropic(create=None, streams=()):
    streams = iter(streams)
    return ns(
        messages=ns(
            create=AsyncMock(side_effect=create or []),
            stream=lambda **kwargs: next(streams),
        )
    )


async def test_anthropic_generate_adapts_response(messages):
    sdk = fake_anthropic(create=[anthropic_message("hola ", "mundo")])
    client = AnthropicClient(make_config("anthropic"), client=sdk)

    response = await client.generate(messages)

    assert response.content == "hola mundo"
    assert (response.provider, response.model, response.finish_reason) == ("anthropic", "claude-test", "end_turn")
    assert (response.usage.input_tokens, response.usage.output_tokens) == (5, 9)
    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 1024
    assert "temperature" not in kwargs


async def test_anthropic_moves_system_message_to_system_param():
    from schemas import ChatMessage

    sdk = fake_anthropic(create=[anthropic_message("ok")])
    msgs = [ChatMessage(role="system", content="Sé breve"), ChatMessage(role="user", content="hola")]
    await AnthropicClient(make_config("anthropic"), client=sdk).generate(msgs)

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["system"] == "Sé breve"
    assert kwargs["messages"] == [{"role": "user", "content": "hola"}]


async def test_anthropic_stream_yields_text(messages):
    sdk = fake_anthropic(streams=[FakeAnthropicStream(agen("La ", "entropía"))])
    client = AnthropicClient(make_config("anthropic"), client=sdk)
    assert await collect(client.stream(messages)) == ["La ", "entropía"]


async def test_anthropic_response_without_text_blocks(messages):
    empty = ns(
        model="m", content=[ns(type="tool_use")], stop_reason="end_turn", usage=ns(input_tokens=1, output_tokens=1)
    )
    client = AnthropicClient(make_config("anthropic"), client=fake_anthropic(create=[empty]))
    with pytest.raises(LLMInvalidResponseError):
        await client.generate(messages)


# ---------- Interfaz común ----------
async def test_both_providers_expose_same_stream_contract(messages):
    openai_c = OpenAIClient(make_config("openai"), client=fake_openai(agen(openai_chunk("a"), openai_chunk("b"))))
    anthropic_c = AnthropicClient(
        make_config("anthropic"), client=fake_anthropic(streams=[FakeAnthropicStream(agen("a", "b"))])
    )
    assert await collect(openai_c.stream(messages)) == await collect(anthropic_c.stream(messages)) == ["a", "b"]
