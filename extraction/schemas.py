"""Esquema Pydantic de la salida estructurada del pipeline de extracción."""

from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


MAX_TECNOLOGIAS = 30

_Tecnologia = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


def _sin_duplicados(items: list[str]) -> list[str]:
    """Elimina repetidos (sin distinguir mayúsculas) conservando el orden de aparición."""
    vistos: set[str] = set()
    unicos: list[str] = []
    for item in items:
        clave = item.casefold()
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(item)
    return unicos


class EntidadesTecnicas(BaseModel):
    """Resultado de analizar un texto técnico (descripción de arquitectura, log de error, etc.)."""

    # Campos extra se rechazan: una respuesta con claves inventadas no se acepta en silencio.
    model_config = ConfigDict(extra="forbid")

    tecnologias: Annotated[list[_Tecnologia], AfterValidator(_sin_duplicados)] = Field(
        max_length=MAX_TECNOLOGIAS,
        description=(
            "Tecnologías, lenguajes, frameworks, servicios o herramientas mencionados explícitamente en el texto. "
            "Lista vacía si el texto no menciona ninguna."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Criticidad del problema o sistema descrito: 'baja', 'media' o 'alta'."
    )
    resumen_tecnico: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=600)] = Field(
        description="Resumen técnico conciso (1 a 3 oraciones) del contenido del texto.",
    )
