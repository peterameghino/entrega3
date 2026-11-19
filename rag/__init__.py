"""Módulo 3: RAG local sobre runbooks DevOps (data/ -> chunks -> ChromaDB -> LCEL -> respuesta Pydantic)."""

from .chain import RAGPipeline, build_rag_chain, create_rag_from_env, get_rag_response
from .documents import Chunk, discover_documents, load_chunks
from .schemas import MENSAJE_ABSTENCION, Fuente, RespuestaRAG
from .store import Fragmento, LocalEmbeddings, Retriever, ingest

__all__ = [
    "MENSAJE_ABSTENCION",
    "Chunk",
    "Fragmento",
    "Fuente",
    "LocalEmbeddings",
    "RAGPipeline",
    "RespuestaRAG",
    "Retriever",
    "build_rag_chain",
    "create_rag_from_env",
    "discover_documents",
    "get_rag_response",
    "ingest",
    "load_chunks",
]
