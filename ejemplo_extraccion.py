"""Ejemplo ejecutable del pipeline de extracción de entidades técnicas (Módulo 2).

Uso:
    python ejemplo_extraccion.py                       # analiza un log de ejemplo
    python ejemplo_extraccion.py "texto técnico..."    # analiza el texto indicado
    python ejemplo_extraccion.py --archivo log.txt     # analiza el contenido de un archivo
    python ejemplo_extraccion.py --provider openai --model <modelo> "..."

Usa la misma configuración (.env) que main.py. Imprime el resultado validado como JSON.
Códigos de salida: 0 éxito, 2 configuración o entrada inválida, 1 otro error.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from errors import ConfigurationError, InvalidInputError, LLMError
from extraction import create_pipeline_from_env

TEXTO_DE_EJEMPLO = """\
2024-05-14 03:12:07 ERROR [payments-api] psycopg.OperationalError: connection to server at "db-primary" (10.0.3.12),
port 5432 failed: FATAL: remaining connection slots are reserved for non-replication superuser connections
2024-05-14 03:12:07 WARN  [payments-api] Circuit breaker OPEN for PostgreSQL pool (FastAPI worker 4/8)
2024-05-14 03:12:09 ERROR [nginx] upstream timed out (110) while reading response header, upstream: payments-api:8000
2024-05-14 03:12:10 INFO  [k8s] HPA scaled payments-api from 8 to 16 replicas (CPU 91%)
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae tecnologías, criticidad y resumen de un texto técnico.")
    parser.add_argument("texto", nargs="?", help="Texto técnico a analizar (por defecto, un log de ejemplo)")
    parser.add_argument("--archivo", type=Path, help="Leer el texto desde un archivo UTF-8")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="Sobrescribe LLM_PROVIDER")
    parser.add_argument("--model", help="Sobrescribe LLM_MODEL")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    try:
        texto = args.archivo.read_text(encoding="utf-8") if args.archivo else (args.texto or TEXTO_DE_EJEMPLO)
    except OSError as exc:
        print(f"No se pudo leer el archivo: {exc.strerror}", file=sys.stderr)
        return 2
    try:
        pipeline = create_pipeline_from_env(args.provider, args.model)
        resultado = await pipeline.process_text(texto)
    except (ConfigurationError, InvalidInputError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    print(resultado.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    # En consolas Windows la codificación por defecto no es UTF-8: evita acentos corruptos en el JSON.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
