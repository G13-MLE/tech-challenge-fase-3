"""Schemas Pydantic para request e response da API."""

from pydantic import BaseModel, Field, field_validator

from src.models.urgency import UrgencyLevel

__all__ = ["PredictRequest", "PredictResponse"]


class PredictRequest(BaseModel):
    """Schema de requisição para classificação de urgência.

    Attributes:
        text: Texto do laudo médico a ser classificado.
    """

    text: str = Field(..., min_length=1, max_length=10_000, description="Texto do laudo médico")

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Remove espaços desnecessários e rejeita texto vazio após strip.

        Args:
            v: Texto bruto do request.

        Returns:
            Texto com espaços removidos nas extremidades.

        Raises:
            ValueError: Se o texto for vazio após remoção de espaços.
        """
        v = v.strip()
        if not v:
            raise ValueError("Texto não pode ser vazio ou apenas espaços")
        return v


class PredictResponse(BaseModel):
    """Schema de resposta com o resultado da classificação.

    Attributes:
        urgency: Nível de urgência (normal, atenção, urgente).
        confidence: Probabilidade associada à predição (0.0 a 1.0).
    """

    urgency: UrgencyLevel = Field(description="Nível de urgência: normal, atenção ou urgente")
    confidence: float = Field(ge=0.0, le=1.0, description="Probabilidade da predição")
