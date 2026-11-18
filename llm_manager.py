from collections.abc import AsyncIterator, Sequence

from clients.anthropic_client import AnthropicClient
from clients.base import BaseLLMClient
from clients.openai_client import OpenAIClient
from config import load_config
from errors import ConfigurationError
from schemas import ChatMessage, ModelConfig, ModelResponse

_CLIENTS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


def create_client(config: ModelConfig) -> BaseLLMClient:
    """Factory: elige la implementación según config.provider."""
    try:
        client_cls = _CLIENTS[config.provider]
    except KeyError:
        raise ConfigurationError(f"Proveedor no soportado: {config.provider}") from None
    return client_cls(config)


class AsyncLLMManager:
    """Punto de entrada para el código de negocio: no conoce ningún SDK."""

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    @classmethod
    def from_config(cls, config: ModelConfig) -> "AsyncLLMManager":
        """Crea el manager con el cliente del proveedor indicado en `config`."""
        return cls(create_client(config))

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> "AsyncLLMManager":
        """Crea el manager desde variables de entorno. Lanza ConfigurationError si faltan o son inválidas."""
        return cls.from_config(load_config(provider, model))

    @property
    def config(self) -> ModelConfig:
        """Configuración activa del cliente."""
        return self._client.config

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Respuesta completa, con reintentos ante errores transitorios. Lanza errores de `errors`."""
        return await self._client.generate(messages)

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Fragmentos de texto a medida que llegan. Solo reintenta antes del primer fragmento."""
        return self._client.stream(messages)
