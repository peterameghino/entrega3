"""Ingesta de documentos del RAG (Módulo 3): data/ -> limpieza -> chunks de 500 tokens -> ChromaDB local.

Uso:
    python ingestar.py                  # indexa (o sincroniza) los documentos de data/ en chroma_db/
    python ingestar.py --reconstruir    # borra la colección y la vuelve a crear desde cero

Es idempotente: ejecutarla otra vez sin cambios en data/ no agrega ni recalcula nada.
No usa API keys: los embeddings se calculan en local (la primera vez se descarga el modelo).
Códigos de salida: 0 éxito, 2 error de la base de conocimiento (sin documentos, índice incompatible), 1 otro error.
"""

import argparse
import os
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings

from errors import KnowledgeBaseError
from rag.documents import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from rag.store import CHROMA_DIR, COLLECTION_NAME, LocalEmbeddings, ingest, model_name_of


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indexa los documentos de data/ en ChromaDB (local y persistente).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Carpeta de documentos (por defecto data/)")
    parser.add_argument("--db-dir", type=Path, default=CHROMA_DIR, help="Carpeta del índice (por defecto chroma_db/)")
    parser.add_argument("--reconstruir", action="store_true", help="Borra la colección antes de indexar")
    return parser.parse_args(argv)


def mostrar(path: Path) -> str:
    """Ruta relativa al directorio actual cuando es posible (la salida no expone rutas personales)."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None, embeddings: Embeddings | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # consolas Windows: acentos correctos también en los errores
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args = parse_args(argv)
    embeddings = embeddings or LocalEmbeddings()
    print(f"Documentos: {mostrar(args.data_dir)}")
    print(f"Chunking: {CHUNK_SIZE} tokens, solapamiento {CHUNK_OVERLAP} | Embeddings: {model_name_of(embeddings)}")
    print(
        "Cargando el modelo de embeddings (la primera vez se descarga, ~640 MB; puede tardar unos minutos)...",
        flush=True,
    )
    try:
        report = ingest(embeddings, data_dir=args.data_dir, db_dir=args.db_dir, rebuild=args.reconstruir)
    except KnowledgeBaseError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    for documento, chunks in report.documentos.items():
        print(f"  {documento}: {chunks} chunk(s)")
    print(
        f"Colección '{COLLECTION_NAME}' en {mostrar(args.db_dir)}: {report.total} chunks "
        f"(nuevos {report.nuevos}, actualizados {report.actualizados}, sin cambios {report.sin_cambios}, "
        f"eliminados {report.eliminados})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
