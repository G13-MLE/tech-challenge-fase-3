"""Aplicação FastAPI para triagem de urgência de laudos médicos.

Expõe os endpoints /health e /predict, carregando o modelo no
startup via Factory Pattern (joblib).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schemas import PredictRequest, PredictResponse
from src.core.config import get_settings
from src.models.factory import ModelLoader, create_model

HEALTH_OK = "ok"

_model: ModelLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo no startup e libera recursos no shutdown."""
    global _model
    settings = get_settings()
    _model = create_model(settings.model_path, backend=settings.model_backend)
    yield
    _model = None


app = FastAPI(
    title="triage-api",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check da API.

    Returns:
        Dicionário com status da aplicação.
    """
    return {"status": HEALTH_OK}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classifica o nível de urgência de um laudo médico.

    Args:
        request: Requisição com o texto do laudo.

    Returns:
        Predição com nível de urgência e confiança.

    Raises:
        RuntimeError: Se o modelo não estiver carregado.
    """
    if _model is None:
        raise RuntimeError("Modelo não carregado.")
    urgency, confidence = _model.predict(request.text)
    return PredictResponse(urgency=urgency, confidence=confidence)
