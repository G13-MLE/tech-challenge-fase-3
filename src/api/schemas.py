"""Schemas Pydantic para request e response da API."""

from pydantic import BaseModel, Field

from src.models.urgency import UrgencyLevel


class PredictRequest(BaseModel):
    """Schema de requisição para classificação de urgência.

    Attributes:
        text: Texto do laudo médico a ser classificado.
    """

    text: str = Field(..., min_length=1, description="Texto do laudo médico")


class PredictResponse(BaseModel):
    """Schema de resposta com o resultado da classificação.

    Attributes:
        urgency: Nível de urgência (normal, atenção, urgente).
        confidence: Probabilidade associada à predição (0.0 a 1.0).
    """

    urgency: UrgencyLevel = Field(description="Nível de urgência: normal, atenção ou urgente")
    confidence: float = Field(ge=0.0, le=1.0, description="Probabilidade da predição")
