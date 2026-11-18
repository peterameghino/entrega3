"""Adaptador entre `ModelConfig` (Módulo 1) y los chat models oficiales de LangChain (Módulo 2).

Reutiliza la configuración y la jerarquía de errores existentes: no hay una segunda carga de configuración.
Los chat models se crean con `max_retries=0` (igual que los SDKs en el Módulo 1): el único nivel de reintentos
es el del pipeline LCEL (`extraction.chain`), así no se multiplican reintentos entre capas.
"""

import anthropic
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from errors import ConfigurationError, LLMError, StructuredOutputError
from schemas import ModelConfig

from .anthropic_client import _translate as _translate_anthropic
from .openai_client import _translate as _translate_openai


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    """Crea el chat model de LangChain del proveedor indicado en `config`."""
    key = config.api_key.get_secret_value()
    if config.provider == "openai":
        return ChatOpenAI(
            model=config.model,
            api_key=SecretStr(key),
            max_completion_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout_s,
            max_retries=0,
        )
    if config.provider == "anthropic":
        # temperature no se pasa: ModelConfig ya la rechaza para Anthropic.
        return ChatAnthropic(
            model=config.model,
            api_key=SecretStr(key),
            max_tokens=config.max_tokens,
            timeout=config.timeout_s,
            max_retries=0,
        )
    raise ConfigurationError(f"Proveedor no soportado: {config.provider}")


def translate_provider_error(exc: Exception, secret: str | None = None) -> LLMError | None:
    """Traduce una excepción de los SDKs de proveedor a `errors.py`. Devuelve None si no es de un SDK.

    Un corte por longitud detectado por el SDK de OpenAI se informa como salida estructurada truncada.
    """
    if isinstance(exc, openai.LengthFinishReasonError):
        return StructuredOutputError("OpenAI cortó la respuesta estructurada por longitud", "truncated")
    if isinstance(exc, openai.OpenAIError):
        return _translate_openai(exc, secret or "")
    if isinstance(exc, anthropic.AnthropicError):
        return _translate_anthropic(exc, secret or "")
    return None
