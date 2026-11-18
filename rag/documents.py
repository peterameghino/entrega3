"""Paso 1 del RAG: leer los documentos de `data/`, limpiarlos y dividirlos en chunks por tokens.

- Fuente de verdad: los archivos `.md` y `.txt` de la carpeta `data/` del proyecto (no recursivo).
- Limpieza: Unicode NFC, saltos de línea uniformes, sin caracteres de control ni espacios sobrantes.
- Chunking: ventanas de 500 tokens con 50 tokens de solapamiento, contadas con el tokenizador `cl100k_base`
  (tiktoken). Los cortes se hacen sobre posiciones de caracteres del texto original, así un chunk nunca parte una
  letra acentuada.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from errors import KnowledgeBaseError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUPPORTED_SUFFIXES = (".md", ".txt")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOKENIZER_NAME = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    """Un fragmento de un documento, con la metadata que se guarda en ChromaDB."""

    documento: str  # nombre del archivo en data/, p. ej. "azure-api-throttling.md"
    titulo: str
    indice: int  # posición del chunk dentro del documento (0, 1, ...)
    total: int
    texto: str
    tokens: int

    @property
    def id(self) -> str:
        """Identificador estable: es también la referencia que cita el modelo (p. ej. "azure-api-throttling.md#0")."""
        return f"{self.documento}#{self.indice}"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.texto.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, str | int]:
        return {
            "documento": self.documento,
            "titulo": self.titulo,
            "fragmento": self.indice,
            "total_fragmentos": self.total,
            "tokens": self.tokens,
            "sha256": self.sha256,
        }


class TokenEncoding(Protocol):
    """Lo que el chunking necesita de un tokenizador (lo cumple `tiktoken.Encoding`)."""

    def encode(self, text: str) -> list[int]: ...

    def decode_with_offsets(self, tokens: list[int]) -> tuple[str, list[int]]: ...


def get_encoding() -> TokenEncoding:
    import tiktoken  # import diferido: la primera carga descarga y cachea la tabla de tokens

    try:
        return tiktoken.get_encoding(TOKENIZER_NAME)
    except Exception as exc:  # sin conexión en la primera ejecución
        raise KnowledgeBaseError(
            f"No se pudo cargar el tokenizador {TOKENIZER_NAME} ({type(exc).__name__}). La primera ejecución descarga "
            "su tabla de tokens (~1.7 MB, sin credenciales): verificar la conexión a internet y volver a ejecutar."
        ) from None


def discover_documents(data_dir: Path = DATA_DIR) -> list[Path]:
    """Archivos `.md`/`.txt` de `data_dir`, ordenados por nombre. Error si no hay ninguno."""
    if not data_dir.is_dir():
        raise KnowledgeBaseError(f"No existe la carpeta de documentos: {data_dir}")
    paths = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise KnowledgeBaseError(f"No hay documentos .md ni .txt en {data_dir}")
    return paths


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏⁠﻿]")


def clean_text(text: str) -> str:
    """Normaliza sin cambiar el contenido: NFC, saltos `\\n`, sin caracteres de control ni espacios o líneas extra."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text.replace("\t", " "))
    lines = [re.sub(r"(?<=\S) {2,}", " ", line).rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_title(text: str, fallback: str) -> str:
    """Primer encabezado Markdown (`# ...`) o, si no hay, la primera línea no vacía."""
    for line in text.splitlines():
        if line.strip():
            return line.lstrip("#").strip()[:200] or fallback
    return fallback


def split_tokens(
    text: str, encoding: TokenEncoding, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[str, int]]:
    """Divide `text` en ventanas de `chunk_size` tokens que se solapan `overlap` tokens.

    Devuelve (texto del chunk, cantidad de tokens). Cada chunk empieza `chunk_size - overlap` tokens después del
    anterior; el último puede ser más corto. Un texto de hasta `chunk_size` tokens produce un único chunk.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError("Se requiere 0 <= overlap < chunk_size")
    tokens = encoding.encode(text)
    if not tokens:
        return []
    _, offsets = encoding.decode_with_offsets(tokens)  # offsets[i] = carácter donde empieza el token i
    chunks: list[tuple[str, int]] = []
    start = 0
    while True:
        end = min(start + chunk_size, len(tokens))
        char_end = offsets[end] if end < len(tokens) else len(text)
        chunks.append((text[offsets[start] : char_end].strip(), end - start))
        if end == len(tokens):
            return chunks
        start = end - overlap


def load_chunks(
    data_dir: Path = DATA_DIR,
    encoding: TokenEncoding | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Lee, limpia y divide todos los documentos de `data_dir`, conservando el nombre de archivo de origen."""
    encoding = encoding or get_encoding()
    chunks: list[Chunk] = []
    for path in discover_documents(data_dir):
        text = clean_text(path.read_text(encoding="utf-8"))
        parts = [(t, n) for t, n in split_tokens(text, encoding, chunk_size=chunk_size, overlap=overlap) if t]
        title = extract_title(text, path.stem)
        chunks += [
            Chunk(documento=path.name, titulo=title, indice=i, total=len(parts), texto=t, tokens=n)
            for i, (t, n) in enumerate(parts)
        ]
    return chunks
