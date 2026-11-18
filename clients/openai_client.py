from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI

from errors import (
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
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
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError(f"OpenAI timeout: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return LLMConnectionError(f"OpenAI connection error: {exc}")
    if isinstance(exc, openai.APIStatusError):
        retry_after = parse_retry_after(exc.response.headers)
        error = error_from_status(exc.status_code, f"OpenAI HTTP {exc.status_code}: {exc.message}", retry_after)
        if isinstance(error, LLMRateLimitError) and "insufficient_quota" in str(exc.body):
            # 429 por cuota agotada: reintentar no ayuda.
            return LLMBadRequestError(f"OpenAI sin cuota disponible: {exc.message}")
        return error
    return LLMBadRequestError(f"OpenAI error inesperado: {type(exc).__name__}")


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, client: AsyncOpenAI | None = None) -> None:
        super().__init__(config)
        self._client = client or AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_s,
            max_retries=0,
        )

    def _params(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            completion = await self._client.chat.completions.create(**self._params(messages))
        except openai.OpenAIError as exc:
            raise _translate(exc, self.config.api_key.get_secret_value()) from exc

        if not completion.choices or completion.choices[0].message.content is None:
            raise LLMInvalidResponseError("OpenAI devolvió una respuesta sin contenido")
        choice = completion.choices[0]
        usage = completion.usage
        return ModelResponse(
            content=choice.message.content,
            provider="openai",
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
            ),
        )

    async def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(**self._params(messages), stream=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as exc:
            raise _translate(exc, self.config.api_key.get_secret_value()) from exc
