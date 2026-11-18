"""Paso 2 del RAG: embeddings locales y almacenamiento persistente en ChromaDB.

- Embeddings: `jinaai/jina-embeddings-v2-base-es` ejecutado en local con fastembed (ONNX, sin API key). Es bilingüe
  español/inglés y admite 8192 tokens, así que un chunk de 500 tokens se representa completo, sin truncar.
- Persistencia: `chromadb.PersistentClient` en `chroma_db/` (separado de `data/`, excluido de Git). Distancia coseno.
- Consistencia: la colección guarda en su metadata el modelo de embeddings, la dimensión y los parámetros de chunking.
  Consultar o reindexar con otra configuración se rechaza con `KnowledgeBaseError` en lugar de mezclar espacios
  vectoriales incompatibles.
- Idempotencia: el id de cada chunk es `<archivo>#<n>`. La ingesta sincroniza la colección con `data/`: agrega lo
  nuevo, re-embebe solo lo que cambió (hash SHA-256), no toca lo que no cambió y borra chunks de documentos eliminados.
"""

import asyncio
import contextlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict

from errors import KnowledgeBaseError

from .documents import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, PROJECT_ROOT, TOKENIZER_NAME, Chunk, load_chunks

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "runbooks"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-es"
# Caché persistente del modelo (fastembed usa por defecto el directorio temporal, que el sistema puede vaciar).
EMBEDDING_CACHE_DIR = Path.home() / ".cache" / "fastembed"


def embedding_cache_dir() -> Path:
    """`FASTEMBED_CACHE_PATH` si está definida; si no, `~/.cache/fastembed`."""
    return Path(os.environ.get("FASTEMBED_CACHE_PATH") or EMBEDDING_CACHE_DIR)


class LocalEmbeddings(Embeddings):
    """Embeddings locales con fastembed, con la interfaz `Embeddings` de LangChain.

    Se usa la MISMA instancia/modelo para indexar (`embed_documents`) y para consultar (`embed_query`). La interfaz
    aporta `aembed_query`, que ejecuta el cálculo en un hilo aparte sin bloquear el event loop.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # import diferido: la primera vez descarga el modelo (~640 MB)

            cache = embedding_cache_dir()
            try:
                self._model = TextEmbedding(self.model_name, cache_dir=str(cache))
            except Exception as exc:  # red, Hugging Face o archivos de caché incompletos
                raise KnowledgeBaseError(
                    f"No se pudo cargar el modelo de embeddings {self.model_name} ({type(exc).__name__}). "
                    "La primera ejecución lo descarga desde Hugging Face (~640 MB, sin credenciales): verificar la "
                    f"conexión a internet y volver a ejecutar. Caché del modelo: {cache}"
                ) from None
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._load().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._load().query_embed([text])))]


def model_name_of(embeddings: Embeddings) -> str:
    """Nombre del modelo de embeddings (se guarda en la colección para detectar incompatibilidades)."""
    return str(getattr(embeddings, "model_name", None) or type(embeddings).__name__)


def open_client(db_dir: Path = CHROMA_DIR) -> ClientAPI:
    return chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))


def _index_config(embeddings: Embeddings, dimension: int) -> dict[str, str | int]:
    return {
        "embedding_model": model_name_of(embeddings),
        "embedding_dim": dimension,
        "tokenizer": TOKENIZER_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def _check_compatible(collection: Collection, expected: dict[str, str | int]) -> None:
    stored = collection.metadata or {}
    diffs = [f"{k}: índice={stored.get(k)!r}, actual={v!r}" for k, v in expected.items() if stored.get(k) != v]
    if diffs:
        raise KnowledgeBaseError(
            "El índice de ChromaDB se creó con otra configuración (" + "; ".join(diffs) + "). "
            "Reconstruirlo con: python ingestar.py --reconstruir"
        )


# --- Ingesta -------------------------------------------------------------------------------------------------------


@dataclass
class IngestReport:
    documentos: dict[str, int] = field(default_factory=dict)  # archivo -> cantidad de chunks
    nuevos: int = 0
    actualizados: int = 0
    sin_cambios: int = 0
    eliminados: int = 0
    total: int = 0


def ingest(
    embeddings: Embeddings,
    *,
    data_dir: Path = DATA_DIR,
    db_dir: Path = CHROMA_DIR,
    rebuild: bool = False,
    chunks: list[Chunk] | None = None,
) -> IngestReport:
    """Indexa los documentos de `data_dir` en la colección persistente. Es idempotente: repetirla no duplica nada."""
    chunks = load_chunks(data_dir) if chunks is None else chunks
    client = open_client(db_dir)
    if rebuild:
        with contextlib.suppress(NotFoundError):
            client.delete_collection(COLLECTION_NAME)

    stored = _existing(client)
    probe = embeddings.embed_query("dimensión")  # fija la dimensión del modelo actual
    expected = _index_config(embeddings, len(probe))
    if stored is not None:
        _check_compatible(stored, expected)
    collection = stored or client.create_collection(
        COLLECTION_NAME,
        embedding_function=None,  # los vectores se calculan acá, siempre con el mismo modelo
        configuration={"hnsw": {"space": "cosine"}},
        metadata=expected,
    )

    current = collection.get(include=["metadatas"])
    previous = {i: (m or {}).get("sha256") for i, m in zip(current["ids"], current["metadatas"] or [], strict=True)}
    by_id = {c.id: c for c in chunks}
    report = IngestReport(documentos={c.documento: c.total for c in chunks})

    stale = sorted(set(previous) - set(by_id))
    if stale:
        collection.delete(ids=stale)
    changed = [c for c in chunks if previous.get(c.id) != c.sha256]
    if changed:
        vectors: list[Sequence[float]] = list(embeddings.embed_documents([c.texto for c in changed]))
        collection.upsert(
            ids=[c.id for c in changed],
            embeddings=vectors,
            documents=[c.texto for c in changed],
            metadatas=[c.metadata() for c in changed],
        )
    report.nuevos = sum(1 for c in changed if c.id not in previous)
    report.actualizados = len(changed) - report.nuevos
    report.sin_cambios = len(chunks) - len(changed)
    report.eliminados = len(stale)
    report.total = collection.count()
    return report


def _existing(client: ClientAPI) -> Collection | None:
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        return None


# --- Recuperación --------------------------------------------------------------------------------------------------


class Fragmento(BaseModel):
    """Un chunk recuperado, con su referencia trazable al documento de `data/`."""

    model_config = ConfigDict(frozen=True)

    id: str
    documento: str
    fragmento: int
    texto: str
    similitud: float  # 1 - distancia coseno (1 = idéntico)


class Retriever:
    """Búsqueda semántica sobre la colección persistente, con el mismo modelo de embeddings que la ingesta.

    Devuelve como máximo `top_k` chunks y descarta los que tienen similitud menor que `min_similarity`, para no
    inyectar en el prompt contexto que no tiene relación con la pregunta.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        db_dir: Path = CHROMA_DIR,
        top_k: int = 4,
        min_similarity: float = 0.30,
    ) -> None:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k debe estar entre 1 y 10")
        self.embeddings = embeddings
        self.db_dir = db_dir
        self.top_k = top_k
        self.min_similarity = min_similarity
        self._collection: Collection | None = None
        self._compatible = False

    def _open(self) -> Collection:
        """Abre la colección; falla antes de cargar el modelo de embeddings si todavía no hubo ingesta."""
        if self._collection is None:
            if not self.db_dir.is_dir():
                raise KnowledgeBaseError(f"No existe el índice en {self.db_dir}. Ejecutar antes: python ingestar.py")
            collection = _existing(open_client(self.db_dir))
            if collection is None or collection.count() == 0:
                raise KnowledgeBaseError("El índice está vacío. Ejecutar antes: python ingestar.py")
            self._collection = collection
        return self._collection

    async def aretrieve(self, pregunta: str) -> list[Fragmento]:
        collection = await asyncio.to_thread(self._open)
        vector = await self.embeddings.aembed_query(pregunta)
        return await asyncio.to_thread(self._search, collection, vector)

    def _search(self, collection: Collection, vector: list[float]) -> list[Fragmento]:
        if not self._compatible:
            _check_compatible(collection, _index_config(self.embeddings, len(vector)))
            self._compatible = True
        result = collection.query(
            query_embeddings=vector,
            n_results=min(self.top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        found = zip(
            result["ids"][0],
            (result["documents"] or [[]])[0],
            (result["metadatas"] or [[]])[0],
            (result["distances"] or [[]])[0],
            strict=True,
        )
        fragmentos = [
            Fragmento(
                id=chunk_id,
                documento=str(meta["documento"]),
                fragmento=int(meta["fragmento"]),  # type: ignore[arg-type]
                texto=text,
                similitud=round(1.0 - float(distance), 4),
            )
            for chunk_id, text, meta, distance in found
        ]
        return [f for f in fragmentos if f.similitud >= self.min_similarity]
