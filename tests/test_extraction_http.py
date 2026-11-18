"""Integración local del pipeline: ChatOpenAI y ChatAnthropic reales contra un servidor HTTP falso en 127.0.0.1.

Ejercita el parsing real de salida estructurada de cada integración oficial, los metadatos de corte por longitud
y que no existan reintentos ocultos en los SDKs (cada respuesta guionada se consume exactamente una vez).
NO prueba las APIs reales de OpenAI/Anthropic.
"""

import asyncio
import json
import logging
from typing import Any

import pytest

from clients.langchain_client import build_chat_model
from errors import LLMAuthenticationError
from extraction import ExtractionPipeline, NivelCriticidad

from .conftest import FAKE_KEY, make_config

GOOD = {
    "tecnologias": ["Kubernetes", "Nginx"],
    "nivel_de_criticidad": "media",
    "resumen_tecnico": "El ingress Nginx devuelve 502 intermitentes en el clúster Kubernetes.",
}
TEXT = "Nginx ingress en Kubernetes devuelve 502 intermitentes desde el último deploy."


def openai_completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def anthropic_tool_use(arguments: dict[str, Any], stop_reason: str = "tool_use") -> dict[str, Any]:
    return {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "model": "cm",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "EntidadesTecnicas", "input": arguments}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class ScriptedServer:
    """Devuelve las respuestas (status, body) guionadas, en orden, y guarda los cuerpos recibidos."""

    def __init__(self, *responses: tuple[int, dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self._server: asyncio.Server | None = None

    async def start(self) -> str:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = (await reader.readuntil(b"\r\n\r\n")).decode().split("\r\n")
        length = next((int(h.split(":")[1]) for h in head if h.lower().startswith("content-length")), 0)
        self.paths.append(head[0].split()[1])
        self.bodies.append(json.loads(await reader.readexactly(length) or b"{}"))
        status, payload = self.responses.pop(0) if self.responses else (500, {"error": {"message": "sin guion"}})
        data = json.dumps(payload).encode()
        writer.write(
            f"HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {len(data)}\r\n"
            "connection: close\r\n\r\n".encode()
            + data
        )
        await writer.drain()
        writer.close()


async def run_pipeline(
    monkeypatch: pytest.MonkeyPatch, provider: str, *responses: tuple[int, dict[str, Any]], max_attempts: int = 3
) -> tuple[Any, ScriptedServer]:
    server = ScriptedServer(*responses)
    url = await server.start()
    if provider == "openai":
        monkeypatch.setenv("OPENAI_BASE_URL", url + "/v1")
    else:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    model = build_chat_model(make_config(provider, max_tokens=256))
    pipeline = ExtractionPipeline(model, max_attempts=max_attempts, secret=FAKE_KEY, wait_max_s=0)
    try:
        return await pipeline.process_text(TEXT), server
    except Exception as exc:
        return exc, server
    finally:
        await server.stop()


async def test_openai_structured_output_retries_invalid_enum_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    invalid = json.dumps({**GOOD, "nivel_de_criticidad": "grave"})
    result, server = await run_pipeline(
        monkeypatch, "openai", (200, openai_completion(invalid)), (200, openai_completion(json.dumps(GOOD)))
    )
    assert result.nivel_de_criticidad is NivelCriticidad.MEDIA
    assert len(server.bodies) == 2
    body = server.bodies[0]
    assert server.paths[0] == "/v1/chat/completions"
    assert body["response_format"]["type"] == "json_schema"
    assert body["max_completion_tokens"] == 256
    assert TEXT in json.dumps(body["messages"], ensure_ascii=False)


async def test_openai_truncated_completion_is_rejected_and_retried(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    truncated = json.dumps(GOOD)[:40]
    caplog.set_level(logging.WARNING, logger="extraction.chain")
    result, server = await run_pipeline(
        monkeypatch,
        "openai",
        (200, openai_completion(truncated, finish_reason="length")),
        (200, openai_completion(json.dumps(GOOD))),
    )
    assert result.tecnologias == ["Kubernetes", "Nginx"]
    assert len(server.bodies) == 2
    assert "tipo=truncated intento=1" in caplog.text


async def test_openai_server_error_is_retried_once_per_attempt_without_sdk_retries(monkeypatch: pytest.MonkeyPatch):
    result, server = await run_pipeline(
        monkeypatch,
        "openai",
        (503, {"error": {"message": "overloaded"}}),
        (200, openai_completion(json.dumps(GOOD))),
    )
    assert result.nivel_de_criticidad is NivelCriticidad.MEDIA
    assert len(server.bodies) == 2


async def test_openai_authentication_error_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    result, server = await run_pipeline(
        monkeypatch,
        "openai",
        (401, {"error": {"message": f"Incorrect API key provided: {FAKE_KEY}", "type": "invalid_request_error"}}),
        (200, openai_completion(json.dumps(GOOD))),
    )
    assert isinstance(result, LLMAuthenticationError)
    assert FAKE_KEY not in str(result)
    assert len(server.bodies) == 1


async def test_anthropic_tool_use_max_tokens_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    partial = {k: v for k, v in GOOD.items() if k != "resumen_tecnico"}
    result, server = await run_pipeline(
        monkeypatch,
        "anthropic",
        (200, anthropic_tool_use(partial, stop_reason="max_tokens")),
        (200, anthropic_tool_use(GOOD)),
    )
    assert result.nivel_de_criticidad is NivelCriticidad.MEDIA
    assert len(server.bodies) == 2
    body = server.bodies[0]
    assert server.paths[0] == "/v1/messages"
    assert body["tools"][0]["name"] == "EntidadesTecnicas"
    assert body["max_tokens"] == 256
