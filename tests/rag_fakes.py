"""Dobles de prueba deterministas del RAG: tokenizador por palabras y embeddings por palabras clave. Sin red."""

import hashlib
import math
import re
import unicodedata

from langchain_core.embeddings import Embeddings


class WordEncoding:
    """Tokenizador falso: un token por palabra (con su espacio siguiente). Cumple `rag.documents.TokenEncoding`."""

    def __init__(self) -> None:
        self._starts: list[int] = []
        self._text = ""

    def encode(self, text: str) -> list[int]:
        self._text = text
        self._starts = [m.start() for m in re.finditer(r"\S+\s*", text)]
        if self._starts:
            self._starts[0] = 0  # el espacio inicial pertenece al primer token
        return list(range(len(self._starts)))

    def decode_with_offsets(self, tokens: list[int]) -> tuple[str, list[int]]:
        return self._text, [self._starts[t] for t in tokens]


def _words(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [w for w in re.findall(r"[a-z0-9]+", plain) if len(w) >= 3]


class KeywordEmbeddings(Embeddings):
    """Embeddings falsos: bolsa de palabras con hashing, normalizada. La similitud coseno refleja palabras comunes."""

    def __init__(self, dimension: int = 256, model_name: str = "fake-keyword-embeddings") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.document_calls = 0
        self.embedded_texts = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in _words(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        self.embedded_texts += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
