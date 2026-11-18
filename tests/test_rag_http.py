"""Integración local del RAG: ChatAnthropic y ChatOpenAI reales contra un servidor HTTP falso en 127.0.0.1.

Ejercita el parsing real de la salida estructurada `RespuestaGenerada` en cada integración oficial, con el índice
ChromaDB real (embeddings falsos). Cada respuesta guionada se consume exactamente una vez: no hay reintentos ocultos.
NO prueba las APIs reales.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.client import SharedSystemClient

from clients.langchain_client import build_chat_model
from errors import LLMAuthenticationError
from rag import MENSAJE_ABSTENCION, RAGPipeline
from rag.documents import DATA_DIR, load_chunks
from rag.store import Retriever, ingest

from .conftest import FAKE_KEY, make_config
from .rag_fakes import KeywordEmbeddings, WordEncoding
from .test_extraction_http import ScriptedServer, openai_completion

QUESTION = "¿Qué hago si la API de Azure devuelve 429 por throttling? ¿Respeto Retry-After?"
GROUNDED = {
    "respuesta": "Respetar Retry-After y aplicar backoff exponencial con jitter.",
    "evidencia_suficiente": True,
    "fuentes": ["azure-api-throttling.md#0"],
    "aspectos_sin_respaldo": [],
}


def anthropic_answer(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "model": "cm",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "RespuestaGenerada", "input": arguments}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("rag_http") / "chroma_db"
    ingest(KeywordEmbeddings(), db_dir=path, chunks=load_chunks(DATA_DIR, WordEncoding()))
    yield path
    SharedSystemClient.clear_system_cache()


async def run_rag(
    monkeypatch: pytest.MonkeyPatch, db: Path, provider: str, *responses: tuple[int, dict[str, Any]]
) -> tuple[Any, ScriptedServer]:
    server = ScriptedServer(*responses)
    url = await server.start()
    if provider == "openai":
        monkeypatch.setenv("OPENAI_BASE_URL", url + "/v1")
    else:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    model = build_chat_model(make_config(provider, max_tokens=256))
    retriever = Retriever(KeywordEmbeddings(), db_dir=db, top_k=3, min_similarity=0.1)
    pipeline = RAGPipeline(retriever, model, secret=FAKE_KEY, wait_max_s=0)
    try:
        return await pipeline.answer(QUESTION), server
    except Exception as exc:
        return exc, server
    finally:
        await server.stop()


async def test_anthropic_grounded_answer_over_local_http(monkeypatch: pytest.MonkeyPatch, db: Path):
    result, server = await run_rag(monkeypatch, db, "anthropic", (200, anthropic_answer(GROUNDED)))
    assert result.evidencia_suficiente is True
    assert [f.documento for f in result.fuentes] == ["azure-api-throttling.md"]
    assert len(server.bodies) == 1
    body = server.bodies[0]
    assert body["tools"][0]["name"] == "RespuestaGenerada"
    sent = json.dumps(body, ensure_ascii=False)
    assert 'documento=\\"azure-api-throttling.md\\"' in sent  # el contexto recuperado viaja en el prompt
    assert "usando SOLO los fragmentos" in json.dumps(body["system"], ensure_ascii=False)


async def test_openai_abstention_over_local_http(monkeypatch: pytest.MonkeyPatch, db: Path):
    abstained = {**GROUNDED, "respuesta": MENSAJE_ABSTENCION, "evidencia_suficiente": False, "fuentes": []}
    result, server = await run_rag(monkeypatch, db, "openai", (200, openai_completion(json.dumps(abstained))))
    assert result.evidencia_suficiente is False
    assert result.respuesta == MENSAJE_ABSTENCION and result.fuentes == []
    assert server.bodies[0]["response_format"]["type"] == "json_schema"
    assert len(server.bodies) == 1


async def test_hallucinated_citation_is_retried_over_local_http(monkeypatch: pytest.MonkeyPatch, db: Path):
    bad = {**GROUNDED, "fuentes": ["no-existe.md#0"]}
    result, server = await run_rag(
        monkeypatch, db, "anthropic", (200, anthropic_answer(bad)), (200, anthropic_answer(GROUNDED))
    )
    assert result.evidencia_suficiente is True
    assert len(server.bodies) == 2


async def test_auth_error_over_local_http_is_not_retried(monkeypatch: pytest.MonkeyPatch, db: Path):
    error = {"type": "error", "error": {"type": "authentication_error", "message": f"invalid x-api-key {FAKE_KEY}"}}
    result, server = await run_rag(monkeypatch, db, "anthropic", (401, error))
    assert isinstance(result, LLMAuthenticationError)
    assert FAKE_KEY not in str(result)
    assert len(server.bodies) == 1
