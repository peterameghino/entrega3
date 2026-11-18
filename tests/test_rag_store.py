"""RAG, paso 2: ChromaDB persistente, ingesta idempotente, compatibilidad de embeddings y recuperación. Sin red."""

import subprocess
import sys
from pathlib import Path

import pytest
from chromadb.api.client import SharedSystemClient

from errors import KnowledgeBaseError
from rag.documents import DATA_DIR, get_encoding, load_chunks
from rag.store import COLLECTION_NAME, LocalEmbeddings, Retriever, embedding_cache_dir, ingest, open_client

from .rag_fakes import KeywordEmbeddings, WordEncoding


def chunks_of(data_dir: Path = DATA_DIR):
    return load_chunks(data_dir, WordEncoding())


def write_docs(folder: Path, **docs: str) -> Path:
    folder.mkdir(exist_ok=True)
    for name, text in docs.items():
        (folder / name.replace("_", ".")).write_text(text, encoding="utf-8")
    return folder


@pytest.fixture
def db(tmp_path: Path) -> Path:
    yield tmp_path / "chroma_db"
    SharedSystemClient.clear_system_cache()


def test_ingest_indexes_every_chunk_with_metadata(db: Path):
    chunks = chunks_of()
    report = ingest(KeywordEmbeddings(), db_dir=db, chunks=chunks)
    assert report.total == report.nuevos == len(chunks)
    collection = open_client(db).get_collection(COLLECTION_NAME, embedding_function=None)
    stored = collection.get(ids=[chunks[0].id], include=["documents", "metadatas"])
    assert stored["documents"] == [chunks[0].texto]
    assert stored["metadatas"][0]["documento"] == chunks[0].documento
    assert collection.metadata["embedding_model"] == "fake-keyword-embeddings"
    assert collection.metadata["chunk_size"] == 500
    assert collection.metadata["chunk_overlap"] == 50
    assert collection.configuration["hnsw"]["space"] == "cosine"


def test_repeated_ingestion_does_not_duplicate_or_re_embed(db: Path):
    embeddings = KeywordEmbeddings()
    first = ingest(embeddings, db_dir=db, chunks=chunks_of())
    embedded = embeddings.embedded_texts
    second = ingest(embeddings, db_dir=db, chunks=chunks_of())
    assert second.total == first.total
    assert (second.nuevos, second.actualizados, second.eliminados) == (0, 0, 0)
    assert second.sin_cambios == first.total
    assert embeddings.embedded_texts == embedded  # nada se volvió a embeber


def test_ingestion_syncs_changed_and_deleted_documents(tmp_path: Path, db: Path):
    data = write_docs(tmp_path / "data", a_md="# A\n\nuno dos tres", b_md="# B\n\ncuatro cinco")
    ingest(KeywordEmbeddings(), db_dir=db, chunks=chunks_of(data))
    (data / "a.md").write_text("# A\n\nuno dos tres CAMBIADO", encoding="utf-8")
    (data / "b.md").unlink()
    write_docs(data, c_txt="seis siete")
    report = ingest(KeywordEmbeddings(), db_dir=db, chunks=chunks_of(data))
    assert (report.nuevos, report.actualizados, report.eliminados, report.total) == (1, 1, 1, 2)
    ids = open_client(db).get_collection(COLLECTION_NAME, embedding_function=None).get()["ids"]
    assert sorted(ids) == ["a.md#0", "c.txt#0"]


def test_index_persists_across_client_reinitialization_and_processes(db: Path):
    report = ingest(KeywordEmbeddings(), db_dir=db, chunks=chunks_of())
    SharedSystemClient.clear_system_cache()
    assert open_client(db).get_collection(COLLECTION_NAME, embedding_function=None).count() == report.total
    code = (
        "import sys, chromadb; from chromadb.config import Settings; "
        "c = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False)); "
        "print(c.get_collection('runbooks', embedding_function=None).count())"
    )
    out = subprocess.run([sys.executable, "-c", code, str(db)], capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) == report.total


def test_ingest_rejects_a_different_embedding_model_unless_rebuilding(db: Path):
    ingest(KeywordEmbeddings(model_name="modelo-a"), db_dir=db, chunks=chunks_of())
    with pytest.raises(KnowledgeBaseError, match=r"embedding_model.*--reconstruir"):
        ingest(KeywordEmbeddings(model_name="modelo-b"), db_dir=db, chunks=chunks_of())
    report = ingest(KeywordEmbeddings(model_name="modelo-b"), db_dir=db, chunks=chunks_of(), rebuild=True)
    assert report.nuevos == report.total


def test_ingest_rejects_a_different_embedding_dimension(db: Path):
    ingest(KeywordEmbeddings(dimension=64, model_name="m"), db_dir=db, chunks=chunks_of())
    with pytest.raises(KnowledgeBaseError, match="embedding_dim"):
        ingest(KeywordEmbeddings(dimension=32, model_name="m"), db_dir=db, chunks=chunks_of())


async def test_retrieval_returns_relevant_traceable_chunks(db: Path):
    ingest(KeywordEmbeddings(), db_dir=db, chunks=chunks_of())
    retriever = Retriever(KeywordEmbeddings(), db_dir=db, top_k=3, min_similarity=0.0)
    found = await retriever.aretrieve("Azure devuelve 429 throttling: respetar Retry-After y backoff")
    assert 1 <= len(found) <= 3
    assert found[0].documento == "azure-api-throttling.md"
    assert found[0].id == f"{found[0].documento}#{found[0].fragmento}"
    assert found == sorted(found, key=lambda f: -f.similitud)
    readiness = await retriever.aretrieve("readiness probe del pod en AKS sin endpoints")
    assert readiness[0].documento == "aks-readiness-probe.md"


async def test_retrieval_drops_chunks_below_the_similarity_threshold(db: Path):
    ingest(KeywordEmbeddings(), db_dir=db, chunks=chunks_of())
    retriever = Retriever(KeywordEmbeddings(), db_dir=db, top_k=4, min_similarity=0.3)
    assert await retriever.aretrieve("receta paella valenciana azafrán") == []


async def test_query_with_another_embedding_model_is_rejected(db: Path):
    ingest(KeywordEmbeddings(model_name="modelo-a"), db_dir=db, chunks=chunks_of())
    retriever = Retriever(KeywordEmbeddings(model_name="modelo-b"), db_dir=db)
    with pytest.raises(KnowledgeBaseError, match="otra configuración"):
        await retriever.aretrieve("pregunta")


async def test_query_without_index_is_a_knowledge_base_error(tmp_path: Path, db: Path):
    with pytest.raises(KnowledgeBaseError, match="ingestar"):
        await Retriever(KeywordEmbeddings(), db_dir=tmp_path / "no-existe").aretrieve("pregunta")
    open_client(db)  # carpeta creada pero sin colección
    with pytest.raises(KnowledgeBaseError, match="ingestar"):
        await Retriever(KeywordEmbeddings(), db_dir=db).aretrieve("pregunta")


@pytest.mark.parametrize("top_k", [0, 11])
def test_top_k_is_bounded(top_k: int):
    with pytest.raises(ValueError):
        Retriever(KeywordEmbeddings(), top_k=top_k)


def test_local_embeddings_use_one_model_for_documents_and_queries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    loaded: list[str] = []
    caches: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, name: str, cache_dir: str) -> None:
            loaded.append(name)
            caches.append(cache_dir)

        def passage_embed(self, texts):
            import numpy as np

            return (np.array([1.0, 0.0]) for _ in texts)

        def query_embed(self, texts):
            import numpy as np

            return (np.array([0.0, 1.0]) for _ in texts)

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "cache"))
    embeddings = LocalEmbeddings()
    assert embeddings.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert embeddings.embed_query("q") == [0.0, 1.0]
    assert loaded == ["jinaai/jina-embeddings-v2-base-es"]  # cargado una sola vez, mismo modelo
    assert caches == [str(tmp_path / "cache")]  # FASTEMBED_CACHE_PATH tiene prioridad


def test_default_model_cache_is_persistent_not_the_temp_dir(monkeypatch: pytest.MonkeyPatch):
    import tempfile

    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    cache = embedding_cache_dir()
    assert cache == Path.home() / ".cache" / "fastembed"
    assert not cache.is_relative_to(tempfile.gettempdir())


def test_model_download_failure_is_an_actionable_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def offline(*args, **kwargs):
        raise OSError("connection refused")

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", offline)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    with pytest.raises(KnowledgeBaseError, match="verificar la conexión") as info:
        LocalEmbeddings().embed_query("q")
    assert "640 MB" in str(info.value) and str(tmp_path) in str(info.value)


def test_tokenizer_download_failure_is_an_actionable_error(monkeypatch: pytest.MonkeyPatch):
    import tiktoken

    def offline(name: str):
        raise OSError("connection refused")

    monkeypatch.setattr(tiktoken, "get_encoding", offline)
    with pytest.raises(KnowledgeBaseError, match="verificar la conexión"):
        get_encoding()


class NoEmbeddings(KeywordEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("no debe cargar el modelo sin índice")


async def test_query_without_index_fails_before_loading_the_embedding_model(tmp_path: Path, db: Path):
    with pytest.raises(KnowledgeBaseError, match="ingestar"):
        await Retriever(NoEmbeddings(), db_dir=tmp_path / "no-existe").aretrieve("pregunta")
    open_client(db)
    with pytest.raises(KnowledgeBaseError, match="ingestar"):
        await Retriever(NoEmbeddings(), db_dir=db).aretrieve("pregunta")
