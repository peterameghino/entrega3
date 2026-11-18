"""Jerarquía de errores propia, independiente de los SDKs."""

import re


class LLMError(Exception):
    """Base de todos los errores del cliente."""


class ConfigurationError(LLMError):
    """Configuración inválida o faltante. Permanente."""


class RetryableError(LLMError):
    """Error transitorio: tiene sentido reintentar."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMConnectionError(RetryableError):
    """Fallo de red o servicio no disponible (5xx)."""


class LLMTimeoutError(RetryableError):
    """La solicitud excedió el timeout."""


class LLMRateLimitError(RetryableError):
    """HTTP 429."""


class LLMAuthenticationError(LLMError):
    """API key inválida o sin permisos (401/403). Permanente."""


class LLMBadRequestError(LLMError):
    """Solicitud rechazada por el proveedor (4xx). Permanente."""


class LLMInvalidResponseError(LLMError):
    """La respuesta del proveedor no tiene el formato esperado. Permanente."""


class InvalidInputError(LLMError):
    """Entrada de la aplicación inválida (vacía, demasiado larga). Permanente: reintentar no ayuda."""


class KnowledgeBaseError(LLMError):
    """Base de conocimiento del RAG ausente, vacía o incompatible (otro modelo de embeddings). Permanente."""


class StructuredOutputError(LLMError):
    """La respuesta del modelo no cumple el esquema pedido. Reintentable por el pipeline de extracción.

    `kind` distingue la causa: malformed, missing_fields, invalid_enum, invalid_value, truncated, empty.
    """

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def error_from_status(status: int, message: str, retry_after: float | None = None) -> LLMError:
    """Traduce un status HTTP a un error propio."""
    if status == 429:
        return LLMRateLimitError(message, retry_after)
    if status in (401, 403):
        return LLMAuthenticationError(message)
    if status >= 500:
        return LLMConnectionError(message, retry_after)
    return LLMBadRequestError(message)


def parse_retry_after(headers: object) -> float | None:
    """Lee el header Retry-After (segundos). Ignora formatos de fecha."""
    try:
        return float(headers.get("retry-after"))  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return None


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(authorization|x-api-key|api-key)\b\s*[:=]\s*\S+"),
]


def redact(text: str, secret: str | None = None) -> str:
    """Quita de un mensaje la API key configurada y patrones comunes de credenciales."""
    if secret:
        text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def redacted(error: LLMError, secret: str | None = None) -> LLMError:
    """Devuelve el mismo error con el mensaje saneado."""
    error.args = (redact(str(error), secret),)
    return error
