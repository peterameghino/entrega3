from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

Provider = Literal["openai", "anthropic"]
Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelResponse(BaseModel):
    content: str
    provider: Provider
    model: str
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)


class ModelConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    api_key: SecretStr = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    timeout_s: float = Field(default=30.0, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay_s: float = Field(default=0.5, ge=0, le=30)

    @model_validator(mode="after")
    def _check_provider_constraints(self) -> Self:
        # El SDK de Anthropic (2.x) no expone `temperature` en messages.create/stream.
        if self.provider == "anthropic" and self.temperature is not None:
            raise ValueError("temperature no está soportado para el proveedor 'anthropic'")
        return self
