import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from errors import RetryableError
from schemas import ChatMessage, ModelConfig, ModelResponse

logger = logging.getLogger(__name__)

MAX_RETRY_DELAY_S = 30.0


class BaseLLMClient(ABC):
    """Interfaz común. Los reintentos viven acá y no en los SDKs (max_retries=0 en ambos),
    para que haya un único nivel de retry con la misma política para todos los proveedores.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        attempt = 0
        while True:
            try:
                return await self._generate_once(messages)
            except RetryableError as exc:
                await self._wait_before_retry(exc, attempt)
                attempt += 1

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Devuelve fragmentos de texto. Solo se reintenta si todavía no se emitió nada:
        reintentar a mitad del stream duplicaría contenido ya entregado al consumidor.
        """
        attempt = 0
        while True:
            emitted = False
            try:
                async for chunk in self._stream_once(messages):
                    emitted = True
                    yield chunk
                return
            except RetryableError as exc:
                if emitted:
                    raise
                await self._wait_before_retry(exc, attempt)
                attempt += 1

    async def _wait_before_retry(self, exc: RetryableError, attempt: int) -> None:
        if attempt >= self.config.max_retries:
            raise exc
        delay = exc.retry_after
        if delay is None:
            delay = self.config.retry_base_delay_s * 2**attempt
        delay = min(delay, MAX_RETRY_DELAY_S)
        logger.warning(
            "%s: %s. Reintento %d/%d en %.1fs",
            self.config.provider,
            type(exc).__name__,
            attempt + 1,
            self.config.max_retries,
            delay,
        )
        await asyncio.sleep(delay)

    @abstractmethod
    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Una llamada al proveedor. Debe lanzar errores de `errors`, nunca del SDK."""

    @abstractmethod
    def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Un stream del proveedor. Debe lanzar errores de `errors`, nunca del SDK."""
