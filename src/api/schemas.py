"""Schemas Pydantic para request e response da API."""

import re

from pydantic import BaseModel, Field, field_validator

from src.models.urgency import UrgencyLevel

__all__ = ["PredictRequest", "PredictBatchRequest", "PredictResponse"]

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class PredictRequest(BaseModel):
    """Schema de requisição para classificação de urgência.

    Attributes:
        text: Texto do laudo médico a ser classificado.
    """

    text: str = Field(..., min_length=1, max_length=10_000, description="Texto do laudo médico")

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Remove espaços desnecessários, caracteres de controle e rejeita vazio.

        Args:
            v: Texto bruto do request.

        Returns:
            Texto com espaços removidos nas extremidades e sem caracteres de controle.

        Raises:
            ValueError: Se o texto for vazio após remoção de espaços.
        """
        v = _CONTROL_CHARS_RE.sub(" ", v)
        v = re.sub(r"\s+", " ", v).strip()
        if not v:
            raise ValueError("Texto não pode ser vazio ou apenas espaços")
        return v


class PredictBatchRequest(BaseModel):
    """Schema de requisição para classificação em lote.

    Attributes:
        texts: Lista de textos de laudos médicos (1 a 100).
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de textos de laudos médicos",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        """Aplica as validações de texto a cada item do lote.

        Args:
            v: Lista de textos brutos.

        Returns:
            Lista de textos limpos.

        Raises:
            ValueError: Se algum item for vazio ou exceder o tamanho máximo.
        """
        cleaned: list[str] = []
        for text in v:
            text = _CONTROL_CHARS_RE.sub(" ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                raise ValueError("Texto vazio não permitido no lote")
            if len(text) > 10_000:
                raise ValueError("Texto excede o tamanho máximo de 10000 caracteres")
            cleaned.append(text)
        return cleaned


class PredictResponse(BaseModel):
    """Schema de resposta com o resultado da classificação.

    Attributes:
        urgency: Nível de urgência (normal, atenção, urgente).
        confidence: Probabilidade associada à predição (0.0 a 1.0).
        model_version: Identificador da versão do modelo (prefixo do hash SHA256).
    """

    urgency: UrgencyLevel = Field(description="Nível de urgência: normal, atenção ou urgente")
    confidence: float = Field(ge=0.0, le=1.0, description="Probabilidade da predição")
    model_version: str = Field(default="", description="Identificador da versão do modelo")
