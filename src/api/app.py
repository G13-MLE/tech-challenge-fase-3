"""Aplicação FastAPI para triagem de urgência de laudos médicos.

Expõe os endpoints /health e /predict, carregando o modelo no
startup via Factory Pattern (joblib) com verificação de integridade.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.schemas import PredictRequest, PredictResponse
from src.core.config import get_settings
from src.models.factory import ModelLoader, create_model
from src.preprocess.normalizer import (
    LowercaseNormalizer,
    PunctuationNormalizer,
    compose_normalizers,
)

__all__ = ["app"]

logger = logging.getLogger(__name__)

HEALTH_OK = "ok"
HEALTH_DEGRADED = "degraded"


def _build_normalizer():
    """Constrói o normalizador padrão para pré-processamento de texto na inferência.

    Returns:
        TextNormalizer: Estratégia composta (minúsculas sem acento + sem pontuação).
    """
    return compose_normalizers(LowercaseNormalizer(), PunctuationNormalizer())


def get_model(request: Request) -> ModelLoader:
    """Dependency injection para obter o modelo carregado.

    Args:
        request: Request do FastAPI com acesso ao app.state.

    Returns:
        ModelLoader: Instância do modelo carregado.

    Raises:
        RuntimeError: Se o modelo não estiver carregado.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise RuntimeError("Modelo não carregado.")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo no startup e libera recursos no shutdown."""
    settings = get_settings()
    try:
        app.state.model = create_model(settings.model_path, backend=settings.model_backend)
        app.state.normalizer = _build_normalizer()
        logger.info("Modelo carregado com sucesso: %s", settings.model_path)
    except Exception:
        logger.exception("Falha ao carregar modelo: %s", settings.model_path)
        app.state.model = None
        app.state.normalizer = None
        raise
    yield
    app.state.model = None
    app.state.normalizer = None


app = FastAPI(
    title="triage-api",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handler para RuntimeError que retorna JSON estruturado.

    Args:
        request: Request do FastAPI.
        exc: Exceção RuntimeError capturada.

    Returns:
        JSONResponse com status 500 e detalhes do erro.
    """
    logger.error("RuntimeError: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Health check da API.

    Verifica se o modelo está carregado e retorna o status adequado.

    Args:
        request: Request do FastAPI para acessar app.state.

    Returns:
        Dicionário com status da aplicação.
    """
    model = getattr(request.app.state, "model", None)
    status = HEALTH_OK if model is not None else HEALTH_DEGRADED
    return {"status": status}


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    model: ModelLoader = Depends(get_model),
) -> PredictResponse:
    """Classifica o nível de urgência de um laudo médico.

    O texto de entrada é normalizado antes da predição para manter
    consistência com o pipeline de treinamento.

    Args:
        request: Requisição com o texto do laudo.
        model: Modelo injetado via dependency injection.

    Returns:
        Predição com nível de urgência e confiança.
    """
    normalizer = getattr(app.state, "normalizer", None)
    if normalizer is not None:
        normalized_text = normalizer.normalize(request.text)
    else:
        normalized_text = request.text
    urgency, confidence = model.predict(normalized_text)
    return PredictResponse(urgency=urgency, confidence=confidence)
