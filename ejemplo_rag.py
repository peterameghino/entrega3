"""Ejemplo ejecutable del RAG (Módulo 3): pregunta -> fragmentos de ChromaDB -> LLM -> respuesta Pydantic validada.

Uso (antes: python ingestar.py):
    python ejemplo_rag.py                       # corre el ejemplo positivo y el negativo
    python ejemplo_rag.py --ejemplo positivo    # pregunta respondida por los runbooks
    python ejemplo_rag.py --ejemplo negativo    # pregunta fuera de contexto: debe abstenerse
    python ejemplo_rag.py "¿Cómo diagnostico un pod que no pasa la readiness probe?"
    python ejemplo_rag.py --provider openai --model <modelo> --top-k 3 "..."

Usa la misma configuración (.env) que main.py y ejemplo_extraccion.py. Imprime cada respuesta validada como JSON.
Códigos de salida: 0 éxito, 2 configuración, entrada o índice inválidos, 1 otro error.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from errors import ConfigurationError, InvalidInputError, KnowledgeBaseError, LLMError
from rag import RAGPipeline, create_rag_from_env
from rag.store import CHROMA_DIR

EJEMPLOS = {
    # Respondida por data/azure-api-throttling.md.
    "positivo": "¿Qué debo hacer cuando una automatización recibe HTTP 429 de la API de Azure?",
    # Tema DevOps, pero ningún documento de data/ lo cubre: la respuesta correcta es abstenerse.
    "negativo": "¿Cómo configuro la replicación de Apache Kafka entre dos regiones?",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Responde preguntas con los runbooks indexados en ChromaDB.")
    parser.add_argument("pregunta", nargs="?", help="Pregunta (por defecto, los ejemplos positivo y negativo)")
    parser.add_argument("--ejemplo", choices=sorted(EJEMPLOS), help="Corre solo uno de los ejemplos")
    parser.add_argument("--top-k", type=int, default=4, help="Fragmentos a recuperar, 1 a 10 (por defecto 4)")
    parser.add_argument("--db-dir", type=Path, default=CHROMA_DIR, help="Carpeta del índice (por defecto chroma_db/)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="Sobrescribe LLM_PROVIDER")
    parser.add_argument("--model", help="Sobrescribe LLM_MODEL")
    return parser.parse_args(argv)


def preguntas(args: argparse.Namespace) -> list[str]:
    if args.pregunta:
        return [args.pregunta]
    if args.ejemplo:
        return [EJEMPLOS[args.ejemplo]]
    return [EJEMPLOS["positivo"], EJEMPLOS["negativo"]]


async def run(args: argparse.Namespace, pipeline: RAGPipeline | None = None) -> int:
    try:
        if pipeline is None:
            if not 1 <= args.top_k <= 10:
                raise InvalidInputError("--top-k debe estar entre 1 y 10")
            pipeline = create_rag_from_env(args.provider, args.model, top_k=args.top_k, db_dir=args.db_dir)
        for pregunta in preguntas(args):
            respuesta = await pipeline.answer(pregunta)
            print(respuesta.model_dump_json(indent=2))
    except (ConfigurationError, InvalidInputError, KnowledgeBaseError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None, pipeline: RAGPipeline | None = None) -> int:
    # En consolas Windows la codificación por defecto no es UTF-8: evita acentos corruptos en el JSON y los errores.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(run(parse_args(argv), pipeline))


if __name__ == "__main__":
    sys.exit(main())
