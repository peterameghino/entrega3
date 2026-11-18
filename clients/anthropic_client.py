from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from errors import (
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMTimeoutError,
    error_from_status,
    parse_retry_after,
    redacted,
)
from schemas import ChatMessage, ModelConfig, ModelResponse, Usage

from .base import BaseLLMClient


def _translate(exc: Exception, secret: str) -> LLMError:
    return redacted(_classify(exc), secret)


def _classify(exc: Exception) -> LLMError:
    # APITimeoutError hereda de APIConnectionError: va primero.
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError(f"Anthropic timeout: {exc}")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMConnectionError(f"Anthropic connection error: {exc}")
    if isinstance(exc, anthropic.APIStatusError):
        retry_after = parse_retry_after(exc.response.headers)
        # 529 (overloaded) es transitorio y cae en >= 500.
        return error_from_status(exc.status_code, f"Anthropic HTTP {exc.status_code}: {exc.message}", retry_after)
    return LLMBadRequestError(f"Anthropic error inesperado: {type(exc).__name__}")


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, client: AsyncAnthropic | None = None) -> None:
        super().__init__(config)
        self._client = client or AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_s,
            max_retries=0,
        )

    def _params(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        # Anthropic recibe el system prompt como parámetro aparte, no como mensaje.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
            "max_tokens": self.config.max_tokens,
        }
        if system:
            params["system"] = system
        return params

    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            message = await self._client.messages.create(**self._params(messages))
        except anthropic.AnthropicError as exc:
            raise _translate(exc, self.config.api_key.get_secret_value()) from exc

        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise LLMInvalidResponseError("Anthropic devolvió una respuesta sin bloques de texto")
        return ModelResponse(
            content=text,
            provider="anthropic",
            model=message.model,
            finish_reason=message.stop_reason,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
        )

    async def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(**self._params(messages)) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AnthropicError as exc:
            raise _translate(exc, self.config.api_key.get_secret_value()) from exc
