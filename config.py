import os

from dotenv import load_dotenv
from pydantic import ValidationError

from errors import ConfigurationError
from schemas import ModelConfig

_API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_OPTIONAL_VARS = {
    "LLM_TEMPERATURE": "temperature",
    "LLM_MAX_TOKENS": "max_tokens",
    "LLM_TIMEOUT_S": "timeout_s",
    "LLM_MAX_RETRIES": "max_retries",
}


def load_config(provider: str | None = None, model: str | None = None) -> ModelConfig:
    """Arma ModelConfig desde variables de entorno (.env incluido).

    `provider` y `model` permiten sobrescribir LLM_PROVIDER y LLM_MODEL (por ejemplo desde la CLI).
    """
    load_dotenv()
    provider = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if provider not in _API_KEY_VARS:
        raise ConfigurationError(f"LLM_PROVIDER debe ser uno de {sorted(_API_KEY_VARS)}; recibido: {provider!r}")

    model = (model or os.environ.get("LLM_MODEL", "")).strip()
    if not model:
        raise ConfigurationError("Falta el modelo: definir LLM_MODEL (o pasar --model)")

    key_var = _API_KEY_VARS[provider]
    api_key = os.environ.get(key_var, "").strip()
    if not api_key:
        raise ConfigurationError(f"Falta {key_var}: completarla en .env (ver .env.example)")

    values: dict[str, object] = {"provider": provider, "model": model, "api_key": api_key}
    for var, field in _OPTIONAL_VARS.items():
        raw = os.environ.get(var, "").strip()
        if raw:
            values[field] = raw

    try:
        return ModelConfig.model_validate(values)
    except ValidationError as exc:
        # include_input=False evita que la API key aparezca en el mensaje de error.
        details = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'config'}: {e['msg']}" for e in exc.errors(include_input=False)
        )
        raise ConfigurationError(f"Configuración inválida: {details}") from None
