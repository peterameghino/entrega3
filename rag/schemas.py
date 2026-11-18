"""Esquemas Pydantic del RAG: lo que debe devolver el modelo y la respuesta final validada."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MENSAJE_ABSTENCION = "No encuentro información suficiente en los documentos disponibles para responder esta pregunta."

_Texto = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
_Item = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class RespuestaGenerada(BaseModel):
    """Salida estructurada que se le pide al modelo (`with_structured_output`)."""

    model_config = ConfigDict(extra="forbid")

    respuesta: _Texto = Field(
        description="Respuesta en español basada solo en los fragmentos de contexto. "
        "Si no alcanzan, el mensaje de abstención."
    )
    evidencia_suficiente: bool = Field(
        description="true solo si los fragmentos contienen la información necesaria para responder."
    )
    fuentes: list[_Item] = Field(
        max_length=10,
        description="Ids de los fragmentos usados para la respuesta, tal como aparecen en el contexto "
        "(p. ej. 'archivo.md#0'). Vacía si evidencia_suficiente es false.",
    )
    aspectos_sin_respaldo: list[_Item] = Field(
        max_length=10,
        description="Partes de la pregunta que los fragmentos no cubren. Vacía si todo está respaldado.",
    )


class Fuente(BaseModel):
    """Referencia trazable a un chunk de un documento de `data/`."""

    model_config = ConfigDict(frozen=True)

    id: str
    documento: str
    fragmento: int
    similitud: float


class RespuestaRAG(BaseModel):
    """Respuesta final validada.

    Invariantes: si hay evidencia suficiente, cita al menos una fuente; si no, no cita ninguna y la respuesta es la
    abstención. Toda fuente citada pertenece a los fragmentos recuperados (no hay referencias inventadas).
    La validez del esquema no prueba que la respuesta sea correcta: solo que es consistente con lo recuperado.
    """

    model_config = ConfigDict(extra="forbid")

    pregunta: str
    respuesta: _Texto
    evidencia_suficiente: bool
    fuentes: list[Fuente]
    aspectos_sin_respaldo: list[str] = Field(default_factory=list)
    recuperados: list[Fuente] = Field(description="Fragmentos que devolvió la búsqueda semántica, citados o no.")

    @model_validator(mode="after")
    def _consistente(self) -> Self:
        if self.evidencia_suficiente and not self.fuentes:
            raise ValueError("una respuesta con evidencia suficiente debe citar al menos una fuente")
        if not self.evidencia_suficiente and (self.fuentes or self.respuesta != MENSAJE_ABSTENCION):
            raise ValueError("una abstención no cita fuentes y usa el mensaje de abstención")
        recuperados = {f.id for f in self.recuperados}
        if any(f.id not in recuperados for f in self.fuentes):
            raise ValueError("todas las fuentes citadas deben estar entre los fragmentos recuperados")
        return self
