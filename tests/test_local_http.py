"""Integración local: SDKs reales contra un servidor HTTP falso en 127.0.0.1.

Valida el parsing real de respuestas JSON y SSE, y el mapeo de errores HTTP, sin red externa ni
API keys reales. NO prueba las APIs de OpenAI/Anthropic.
"""

import asyncio
import json

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from clients.anthropic_client import AnthropicClient
from clients.openai_client import OpenAIClient
from errors import LLMAuthenticationError, LLMConnectionError

from .conftest import collect, make_config

WORDS = ["La ", "entropía ", "mide desorden"]


def sse(events: list[tuple[str | None, object]]) -> str:
    out = ""
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        out += (f"event: {name}\n" if name else "") + f"data: {payload}\n\n"
    return out


def openai_chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


OPENAI_JSON = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "respuesta normal"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}
OPENAI_SSE = sse(
    [(None, openai_chunk({"role": "assistant", "content": w})) for w in WORDS]
    + [(None, openai_chunk({}, "stop")), (None, "[DONE]")]
)

ANTHROPIC_JSON = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": "cm",
    "content": [{"type": "text", "text": "respuesta normal"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 4, "output_tokens": 2},
}
ANTHROPIC_SSE = sse(
    [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "model": "cm",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    + [
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": w}})
        for w in WORDS
    ]
    + [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
)


class FakeServer:
    """Servidor mínimo: responde según el path. `rate_limit_first` simula un 429 y luego éxito."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.port = 0
        self.truncated_sse = ""  # payload SSE incompleto para paths /trunc: se anuncia más largo de lo enviado
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = (await reader.readuntil(b"\r\n\r\n")).decode().split("\r\n")
        path = head[0].split()[1]
        length = next((int(h.split(":")[1]) for h in head if h.lower().startswith("content-length")), 0)
        body = json.loads(await reader.readexactly(length) or b"{}")
        self.requests.append(path)

        status, ctype, payload, extra = 200, "application/json", "", ""
        if path.startswith("/auth"):
            status, payload = 401, json.dumps({"error": {"message": "bad key", "type": "authentication_error"}})
        elif path.startswith("/rl") and self.requests.count(path) == 1:
            status, extra = 429, "retry-after: 0\r\n"
            payload = json.dumps({"error": {"message": "slow down", "type": "rate_limit"}})
        elif path.startswith("/trunc"):
            data = self.truncated_sse.encode()
            head = f"HTTP/1.1 200 X\r\ncontent-type: text/event-stream\r\ncontent-length: {len(data) + 500}\r\n"
            writer.write(head.encode() + b"connection: close\r\n\r\n" + data)
            await writer.drain()
            writer.close()
            return
        elif path.endswith("/chat/completions"):
            ctype, payload = (
                ("text/event-stream", OPENAI_SSE)
                if body.get("stream")
                else ("application/json", json.dumps(OPENAI_JSON))
            )
        elif path.endswith("/messages"):
            ctype, payload = (
                ("text/event-stream", ANTHROPIC_SSE)
                if body.get("stream")
                else ("application/json", json.dumps(ANTHROPIC_JSON))
            )
        data = payload.encode()
        writer.write(
            f"HTTP/1.1 {status} X\r\ncontent-type: {ctype}\r\ncontent-length: {len(data)}\r\n"
            f"{extra}connection: close\r\n\r\n".encode()
            + data
        )
        await writer.drain()
        writer.close()


@pytest.fixture
async def server():
    srv = FakeServer()
    await srv.start()
    yield srv
    await srv.stop()


def openai_client(base_url: str) -> OpenAIClient:
    sdk = AsyncOpenAI(api_key="test-key-not-real", base_url=base_url, max_retries=0)
    return OpenAIClient(make_config("openai"), client=sdk)


def anthropic_client(base_url: str) -> AnthropicClient:
    sdk = AsyncAnthropic(api_key="test-key-not-real", base_url=base_url, max_retries=0)
    return AnthropicClient(make_config("anthropic"), client=sdk)


async def test_openai_over_local_http(server, messages):
    client = openai_client(server.url + "/v1")
    response = await client.generate(messages)
    assert (response.content, response.usage.input_tokens) == ("respuesta normal", 4)
    assert await collect(client.stream(messages)) == WORDS


async def test_anthropic_over_local_http(server, messages):
    client = anthropic_client(server.url)
    response = await client.generate(messages)
    assert (response.content, response.usage.output_tokens) == ("respuesta normal", 2)
    assert await collect(client.stream(messages)) == WORDS


async def test_rate_limit_then_success_over_local_http(server, messages):
    response = await openai_client(server.url + "/rl/v1").generate(messages)
    assert response.content == "respuesta normal"
    assert server.requests.count("/rl/v1/chat/completions") == 2


async def test_auth_error_over_local_http_is_not_retried(server, messages):
    with pytest.raises(LLMAuthenticationError):
        await openai_client(server.url + "/auth/v1").generate(messages)
    assert server.requests.count("/auth/v1/chat/completions") == 1


async def test_connection_refused_over_local_http(messages):
    with pytest.raises(LLMConnectionError):
        client = anthropic_client("http://127.0.0.1:1")
        client.config = make_config("anthropic", max_retries=0)  # Windows tarda ~2s por intento rechazado
        await client.generate(messages)
