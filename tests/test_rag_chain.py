"""RAG, paso 3: cadena LCEL de recuperación + generación, salida Pydantic, abstención, errores y CLIs.

Usa el índice real de ChromaDB (en tmp) con embeddings falsos y un chat model guionado cuyo camino síncrono falla,
lo que demuestra la ejecución asíncrona. Sin red ni API keys.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.client import SharedSystemClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBranch, RunnableSequence
from langchain_core.runnables.passthrough import RunnableAssign
from pydantic import ValidationError

import ejemplo_rag
import ingestar
from errors import (
    ConfigurationError,
    InvalidInputError,
    KnowledgeBaseError,
    LLMAuthenticationError,
    LLMConnectionError,
    StructuredOutputError,
)
from rag import MENSAJE_ABSTENCION, RAGPipeline, RespuestaRAG, build_rag_chain, create_rag_from_env
from rag.chain import SYSTEM_PROMPT, build_prompt
from rag.documents import DATA_DIR, load_chunks
from rag.schemas import Fuente
from rag.store import Retriever, ingest

from .conftest import FAKE_KEY
from .rag_fakes import KeywordEmbeddings, WordEncoding
from .test_extraction import ScriptedChatModel

POSITIVE = "¿Qué hago si la API de Azure devuelve 429 por throttling? ¿Respeto Retry-After?"
NEGATIVE = "¿Cuál es la receta de la paella valenciana con azafrán?"
THROTTLING = "azure-api-throttling.md#0"


def answer_msg(args: dict[str, Any], **metadata: Any) -> AIMessage:
    call = {"name": "RespuestaGenerada", "args": args, "id": "call_1", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call], response_metadata=metadata)


GROUNDED = {
    "respuesta": "Respetar Retry-After y aplicar backoff exponencial con jitter.",
    "evidencia_suficiente": True,
    "fuentes": [THROTTLING],
    "aspectos_sin_respaldo": [],
}
ABSTAINED = {
    "respuesta": MENSAJE_ABSTENCION,
    "evidencia_suficiente": False,
    "fuentes": [],
    "aspectos_sin_respaldo": ["replicación de Kafka"],
}


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("rag") / "chroma_db"
    ingest(KeywordEmbeddings(), db_dir=path, chunks=load_chunks(DATA_DIR, WordEncoding()))
    yield path
    SharedSystemClient.clear_system_cache()


def make_rag(db: Path, *script: Any, max_attempts: int = 3, min_similarity: float = 0.1):
    model = ScriptedChatModel(script=list(script))
    retriever = Retriever(KeywordEmbeddings(), db_dir=db, top_k=3, min_similarity=min_similarity)
    return RAGPipeline(retriever, model, max_attempts=max_attempts, secret=FAKE_KEY, wait_max_s=0), model


# --- Estructura de la cadena ---------------------------------------------------------------------------------------


def test_chain_is_an_explicit_lcel_composition(db: Path):
    chain = build_rag_chain(Retriever(KeywordEmbeddings(), db_dir=db), ScriptedChatModel(script=[]))
    assert isinstance(chain, RunnableSequence)
    assert isinstance(chain.first, RunnableAssign)  # recuperación: agrega "fragmentos" a la entrada
    branch = chain.last
    assert isinstance(branch, RunnableBranch)
    assert branch.branches[0][1].get_name() == "abstencion_sin_contexto"
    generation = branch.default
    assert isinstance(generation, RunnableSequence)
    assert generation.last.get_name() == "validate_answer"
    assert "ChatPromptTemplate" in repr(generation)
    assert "format_context" in repr(generation)


def test_prompt_enforces_grounding_abstention_and_citations():
    for rule in ("SOLO", "No agregues conocimiento propio", MENSAJE_ABSTENCION, "aspectos_sin_respaldo", "No inventes"):
        assert rule in SYSTEM_PROMPT
    messages = build_prompt().format_messages(contexto="<fragmento id='x#0'>dato</fragmento>", pregunta="¿q?")
    assert [m.type for m in messages] == ["system", "human"]
    assert "<pregunta>\n¿q?\n</pregunta>" in messages[1].content


# --- Camino positivo y abstención ----------------------------------------------------------------------------------


async def test_known_question_produces_a_grounded_answer_with_verified_sources(db: Path):
    pipeline, model = make_rag(db, answer_msg(GROUNDED))
    result = await pipeline.answer(POSITIVE)
    assert isinstance(result, RespuestaRAG)
    assert result.evidencia_suficiente is True
    assert [f.id for f in result.fuentes] == [THROTTLING]
    assert result.fuentes[0].documento == "azure-api-throttling.md"
    assert result.fuentes[0] in result.recuperados
    assert 1 <= len(result.recuperados) <= 3
    assert model.calls == 1
    human = model.prompts[0][-1].content
    assert f'<fragmento id="{THROTTLING}" documento="azure-api-throttling.md">' in human
    assert POSITIVE in human
    assert "Retry-After" in human  # el contexto recuperado llega al modelo


async def test_out_of_context_question_without_relevant_chunks_abstains_without_calling_the_model(db: Path):
    pipeline, model = make_rag(db, AssertionError("no debe llamarse"), min_similarity=0.3)
    result = await pipeline.answer(NEGATIVE)
    assert result.evidencia_suficiente is False
    assert result.respuesta == MENSAJE_ABSTENCION
    assert result.fuentes == [] and result.recuperados == []
    assert model.calls == 0


async def test_model_abstention_with_weak_context_is_preserved_and_cites_nothing(db: Path):
    pipeline, model = make_rag(db, answer_msg({**ABSTAINED, "respuesta": "Kafka se replica con MirrorMaker"}))
    result = await pipeline.answer("¿Cómo replico Kafka entre regiones en el pipeline de despliegue?")
    assert result.evidencia_suficiente is False
    assert result.respuesta == MENSAJE_ABSTENCION  # nunca se muestra un texto no respaldado
    assert result.fuentes == []
    assert result.recuperados  # hubo contexto, pero no alcanzaba
    assert result.aspectos_sin_respaldo == ["replicación de Kafka"]
    assert model.calls == 1


# --- Validación, reintentos y errores ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "kind"),
    [
        ({**GROUNDED, "fuentes": ["inventado.md#9"]}, "invalid_value"),
        ({**GROUNDED, "fuentes": []}, "missing_fields"),
        ({k: v for k, v in GROUNDED.items() if k != "evidencia_suficiente"}, "missing_fields"),
        ({**GROUNDED, "extra": 1}, "invalid_value"),
    ],
)
async def test_invalid_answers_are_retried_then_recovered(db: Path, bad: dict[str, Any], kind: str, caplog):
    pipeline, model = make_rag(db, answer_msg(bad), answer_msg(GROUNDED))
    with caplog.at_level(logging.WARNING):
        result = await pipeline.answer(POSITIVE)
    assert result.evidencia_suficiente and model.calls == 2
    assert f"tipo={kind} intento=1" in caplog.text


async def test_truncated_answer_is_rejected_even_if_it_parses(db: Path):
    pipeline, _model = make_rag(db, answer_msg(GROUNDED, stop_reason="max_tokens"), max_attempts=1)
    with pytest.raises(StructuredOutputError) as info:
        await pipeline.answer(POSITIVE)
    assert info.value.kind == "truncated"


async def test_malformed_response_exhausts_attempts_with_a_controlled_error(db: Path):
    pipeline, model = make_rag(db, AIMessage(content="texto libre sin herramienta"))
    with pytest.raises(StructuredOutputError) as info:
        await pipeline.answer(POSITIVE)
    assert info.value.kind == "empty"
    assert model.calls == 3


async def test_transient_provider_error_is_retried(db: Path):
    pipeline, model = make_rag(db, LLMConnectionError("503"), answer_msg(GROUNDED))
    assert (await pipeline.answer(POSITIVE)).evidencia_suficiente
    assert model.calls == 2


async def test_permanent_provider_error_is_not_retried_and_is_redacted(db: Path):
    pipeline, model = make_rag(db, LLMAuthenticationError(f"401 clave {FAKE_KEY}"))
    with pytest.raises(LLMAuthenticationError):
        await pipeline.answer(POSITIVE)
    assert model.calls == 1


async def test_missing_index_is_not_retried(tmp_path: Path):
    retriever = Retriever(KeywordEmbeddings(), db_dir=tmp_path / "vacio")
    model = ScriptedChatModel(script=[answer_msg(GROUNDED)])
    with pytest.raises(KnowledgeBaseError):
        await RAGPipeline(retriever, model, wait_max_s=0).answer(POSITIVE)
    assert model.calls == 0


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001, None])
async def test_invalid_questions_never_reach_retrieval_or_model(db: Path, question: Any):
    pipeline, model = make_rag(db, answer_msg(GROUNDED))
    with pytest.raises(InvalidInputError):
        await pipeline.answer(question)
    assert model.calls == 0


async def test_logs_and_errors_do_not_leak_question_context_or_key(db: Path, caplog):
    pipeline, _ = make_rag(db, answer_msg({**GROUNDED, "fuentes": ["x#1"]}), max_attempts=2)
    with caplog.at_level(logging.DEBUG), pytest.raises(StructuredOutputError) as info:
        await pipeline.answer(POSITIVE)
    for text in (caplog.text, str(info.value)):
        assert POSITIVE not in text and "Retry-After" not in text and FAKE_KEY not in text


# --- Contrato Pydantic ---------------------------------------------------------------------------------------------


def fuente(i: str) -> Fuente:
    return Fuente(id=i, documento=i.split("#")[0], fragmento=0, similitud=0.5)


@pytest.mark.parametrize(
    "values",
    [
        {"evidencia_suficiente": True, "fuentes": [], "respuesta": "x"},
        {"evidencia_suficiente": False, "fuentes": [fuente("a.md#0")], "respuesta": MENSAJE_ABSTENCION},
        {"evidencia_suficiente": False, "fuentes": [], "respuesta": "inventado"},
        {"evidencia_suficiente": True, "fuentes": [fuente("otro.md#0")], "respuesta": "x"},
    ],
)
def test_response_contract_rejects_inconsistent_answers(values: dict[str, Any]):
    with pytest.raises(ValidationError):
        RespuestaRAG(pregunta="p", recuperados=[fuente("a.md#0")], **values)


# --- CLIs ----------------------------------------------------------------------------------------------------------


@pytest.fixture
def word_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag.documents.get_encoding", WordEncoding)


def test_ingestar_cli_is_idempotent(tmp_path: Path, word_tokens, capsys):
    db = tmp_path / "db"
    assert ingestar.main(["--db-dir", str(db)], embeddings=KeywordEmbeddings()) == 0
    first = capsys.readouterr().out
    assert ingestar.main(["--db-dir", str(db)], embeddings=KeywordEmbeddings()) == 0
    second = capsys.readouterr().out
    assert "azure-api-throttling.md" in first and "nuevos 0, actualizados 0" not in first
    assert "nuevos 0, actualizados 0" in second and "eliminados 0" in second
    SharedSystemClient.clear_system_cache()


def test_ingestar_cli_reports_missing_documents(tmp_path: Path, capsys):
    assert ingestar.main(["--data-dir", str(tmp_path), "--db-dir", str(tmp_path / "db")], KeywordEmbeddings()) == 2
    assert "No hay documentos" in capsys.readouterr().err


async def test_ejemplo_rag_runs_positive_and_negative_examples(db: Path, capsys):
    pipeline, model = make_rag(db, answer_msg(GROUNDED), answer_msg(ABSTAINED))
    assert await ejemplo_rag.run(ejemplo_rag.parse_args([]), pipeline) == 0
    out, decoder, outputs = capsys.readouterr().out.strip(), json.JSONDecoder(), []
    while out:
        obj, end = decoder.raw_decode(out)
        outputs.append(obj)
        out = out[end:].strip()
    assert [o["pregunta"] for o in outputs] == [ejemplo_rag.EJEMPLOS["positivo"], ejemplo_rag.EJEMPLOS["negativo"]]
    assert [o["evidencia_suficiente"] for o in outputs] == [True, False]
    assert model.calls == 2


def test_ejemplo_rag_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys):
    assert ejemplo_rag.main(["--top-k", "0"]) == 2
    assert "top-k" in capsys.readouterr().err
    assert ejemplo_rag.main(["--ejemplo", "positivo"]) == 2  # sin .env: configuración inválida
    assert "LLM_PROVIDER" in capsys.readouterr().err


def test_create_rag_from_env_reuses_module_1_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    with pytest.raises(ConfigurationError):
        create_rag_from_env()
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    pipeline = create_rag_from_env(top_k=2, db_dir=tmp_path, embeddings=KeywordEmbeddings())
    assert pipeline.retriever.top_k == 2
    assert pipeline.retriever.db_dir == tmp_path


# --- Punto de entrada pedido por la consigna: async def get_rag_response(query: str) -------------------------------


@pytest.fixture
def entrypoint(monkeypatch: pytest.MonkeyPatch, db: Path):
    """Sustituye `create_rag_from_env` por el pipeline guionado y registra los argumentos recibidos."""
    import rag.chain

    monkeypatch.setattr(rag.chain, "_shared_embeddings", None)
    state: dict[str, Any] = {"calls": []}

    def install(*script: Any, min_similarity: float = 0.1):
        pipeline, model = make_rag(db, *script, min_similarity=min_similarity)
        state["model"] = model

        def fake_create(*args: Any, **kwargs: Any) -> RAGPipeline:
            state["calls"].append((args, kwargs))
            return pipeline

        monkeypatch.setattr(rag.chain, "create_rag_from_env", fake_create)
        return state

    return install


def test_get_rag_response_matches_the_required_async_signature():
    import inspect

    from rag import get_rag_response as exported
    from rag.chain import get_rag_response

    assert exported is get_rag_response
    assert inspect.iscoroutinefunction(get_rag_response)
    signature = inspect.signature(get_rag_response)
    assert list(signature.parameters) == ["query"]
    assert signature.parameters["query"].annotation is str
    assert signature.return_annotation is RespuestaRAG


async def test_get_rag_response_delegates_to_the_pipeline_and_returns_a_valid_answer(entrypoint):
    from rag import LocalEmbeddings, get_rag_response

    state = entrypoint(answer_msg(GROUNDED), answer_msg(GROUNDED))
    result = await get_rag_response(POSITIVE)
    assert isinstance(result, RespuestaRAG)
    assert result.pregunta == POSITIVE
    assert result.evidencia_suficiente is True
    assert [f.id for f in result.fuentes] == [THROTTLING]
    assert result.fuentes[0] in result.recuperados
    await get_rag_response(POSITIVE)
    assert state["model"].calls == 2
    first, second = (kwargs["embeddings"] for _, kwargs in state["calls"])
    assert isinstance(first, LocalEmbeddings)
    assert first is second  # el modelo de embeddings se comparte entre llamadas, no se vuelve a cargar


async def test_get_rag_response_preserves_abstention(entrypoint):
    from rag import get_rag_response

    state = entrypoint(AssertionError("no debe llamarse"), min_similarity=0.3)
    result = await get_rag_response(NEGATIVE)
    assert result.evidencia_suficiente is False
    assert result.respuesta == MENSAJE_ABSTENCION
    assert result.fuentes == []
    assert state["model"].calls == 0


async def test_get_rag_response_preserves_input_and_provider_errors(entrypoint):
    from rag import get_rag_response

    state = entrypoint(LLMAuthenticationError("401"))
    with pytest.raises(InvalidInputError):  # validada antes de buscar o llamar al modelo
        await get_rag_response("   ")
    assert state["model"].calls == 0
    with pytest.raises(LLMAuthenticationError):  # error permanente: sin reintentos
        await get_rag_response(POSITIVE)
    assert state["model"].calls == 1


async def test_get_rag_response_uses_env_configuration_and_reports_missing_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import rag.chain
    from rag import get_rag_response

    monkeypatch.setattr(rag.chain, "_shared_embeddings", None)
    with pytest.raises(ConfigurationError):  # sin .env: falla antes de buscar o cargar modelos
        await get_rag_response(POSITIVE)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    real_create = rag.chain.create_rag_from_env
    monkeypatch.setattr(
        rag.chain,
        "create_rag_from_env",
        lambda **kw: real_create(db_dir=tmp_path / "sin-indice", embeddings=KeywordEmbeddings()),
    )
    with pytest.raises(KnowledgeBaseError, match="ingestar"):
        await get_rag_response(POSITIVE)
