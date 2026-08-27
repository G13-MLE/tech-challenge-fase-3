"""Aplicação FastAPI para triagem de urgência de laudos médicos.

Expõe os endpoints /health, /metrics, /predict e /predict/batch (versão 1 sob
/api/v1), carregando o modelo no startup via Factory Pattern (joblib)
com verificação de integridade. Inclui middleware de logging, CORS,
rate limit por IP e métricas Prometheus (app_requests_total,
app_request_latency_seconds, predictions_total, prediction_latency_seconds).

A aplicação inicia em modo DEGRADADO se o modelo não puder ser
carregado: /health retorna "degraded" e /predict responde 503.
"""

import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest

from src.api.schemas import PredictBatchRequest, PredictRequest, PredictResponse
from src.core.config import get_settings
from src.core.dataset import MODEL_HASH_PATH
from src.core.metrics import (
    app_request_latency_seconds,
    app_requests_total,
    prediction_latency_seconds,
    predictions_total,
)
from src.core.ratelimit import RateLimiter
from src.models.factory import ModelLoader, create_model
from src.preprocess.normalizer import TextNormalizer, build_default_normalizer

__all__ = ["app"]

logger = logging.getLogger(__name__)

HEALTH_OK = "ok"
HEALTH_DEGRADED = "degraded"
API_PREFIX = "/api/v1"

WARMUP_TEXTS = [
    "paciente com pneumonia bacteriana grave",
    "diabetes tipo 2 descompensada",
    "asma cronica com sibilos",
]


def _warmup_model(model: ModelLoader, normalizer: TextNormalizer) -> None:
    """Executa inferências de warmup após carregar o modelo.

    Realiza predições com textos de exemplo para pré-aquecer caches
    e JIT do runtime. Falhas são apenas registradas em log — não
    impedem a inicialização da API.

    Args:
        model: Modelo carregado.
        normalizer: Normalizador de texto.
    """
    for text in WARMUP_TEXTS:
        try:
            normalized = normalizer.normalize(text)
            if normalized:
                model.predict(normalized)
        except Exception:
            logger.warning("Warmup falhou para texto '%s' (ignorado).", text, exc_info=True)
    logger.info("Warmup concluído (%d textos).", len(WARMUP_TEXTS))


def _read_model_version() -> str:
    """Lê o hash SHA256 do modelo como identificador de versão.

    Returns:
        Prefixo de 16 caracteres do hash SHA256, ou string vazia se indisponível.
    """
    try:
        hash_content = MODEL_HASH_PATH.read_text(encoding="utf-8").strip()
        return hash_content[:16]
    except FileNotFoundError, OSError:
        return ""


def get_model(request: Request) -> ModelLoader:
    """Dependency injection para obter o modelo carregado.

    Args:
        request: Request do FastAPI com acesso ao app.state.

    Returns:
        ModelLoader: Instância do modelo carregado.

    Raises:
        HTTPException: 503 se o modelo não estiver carregado.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não disponível: aplicação em modo degradado.",
        )
    return model


def get_normalizer(request: Request) -> TextNormalizer:
    """Dependency injection para obter o normalizador de texto.

    Args:
        request: Request do FastAPI com acesso ao app.state.

    Returns:
        TextNormalizer: Instância do normalizador.

    Raises:
        HTTPException: 503 se o normalizador não estiver carregado.
    """
    normalizer = getattr(request.app.state, "normalizer", None)
    if normalizer is None:
        raise HTTPException(
            status_code=503,
            detail="Normalizador não disponível no momento.",
        )
    return normalizer


def get_model_version(request: Request) -> str:
    """Dependency injection para obter a versão do modelo.

    Args:
        request: Request do FastAPI com acesso ao app.state.

    Returns:
        Versão do modelo (prefixo do hash SHA256).
    """
    return getattr(request.app.state, "model_version", "")


def _client_ip(request: Request) -> str:
    """Extrai o IP do cliente do request.

    Args:
        request: Request do FastAPI.

    Returns:
        Endereço IP do cliente (ou 'unknown').
    """
    settings = get_settings()
    if settings.trust_forwarded_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_rate_limiter(request: Request) -> RateLimiter:
    """Retorna o rate limiter armazenado no app.state.

    Args:
        request: Request do FastAPI.

    Returns:
        RateLimiter configurado.
    """
    return request.app.state.rate_limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo no startup e libera recursos no shutdown.

    Se o modelo falhar ao carregar, a aplicação continua em modo
    degradado (model=None, normalizer=None) e o /health reporta
    "degraded".
    """
    settings = get_settings()
    app.state.rate_limiter = RateLimiter(
        max_requests=settings.rate_limit_max_per_ip,
        window_seconds=settings.rate_limit_window_seconds,
    )
    try:
        app.state.model = create_model(settings.model_path, backend=settings.model_backend)
        app.state.normalizer = build_default_normalizer()
        app.state.model_version = _read_model_version()
        logger.info("Modelo carregado com sucesso: %s", settings.model_path)
        _warmup_model(app.state.model, app.state.normalizer)
    except Exception:
        logger.exception("Falha ao carregar modelo: %s", settings.model_path)
        app.state.model = None
        app.state.normalizer = None
        app.state.model_version = ""
        logger.warning("API iniciando em modo DEGRADADO (modelo indisponível).")
    yield
    logger.info("Shutdown: liberando recursos.")
    app.state.model = None
    app.state.normalizer = None
    app.state.model_version = ""


app = FastAPI(
    title="triage-api",
    version="0.1.0",
    lifespan=lifespan,
)


def _configure_cors() -> None:
    """Configura CORS com as origens permitidas das Settings."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


_configure_cors()


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Registra método, path, status e latência de cada requisição."""
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    logger.info(
        "%s %s -> %d (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        latency * 1000,
    )
    app_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=str(response.status_code),
    ).inc()
    app_request_latency_seconds.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=str(response.status_code),
    ).observe(latency)
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Aplica rate limit por IP em endpoints de predição."""
    if request.url.path in {f"{API_PREFIX}/predict", f"{API_PREFIX}/predict/batch"}:
        limiter = _get_rate_limiter(request)
        if not limiter.is_allowed(_client_ip(request)):
            return JSONResponse(
                status_code=429,
                content={"detail": "Limite de requisições excedido. Tente novamente."},
            )
    return await call_next(request)


def _require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Dependency para autenticação via X-API-Key (opcional via Settings).

    Args:
        request: Request do FastAPI.
        x_api_key: Valor do header X-API-Key.

    Raises:
        HTTPException: 401 se a chave for inválida ou ausente.
    """
    settings = get_settings()
    if not settings.api_key_enabled:
        return
    if not secrets.compare_digest(x_api_key or "", settings.api_key.get_secret_value()):
        raise HTTPException(status_code=401, detail="API key inválida ou ausente.")


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


@app.get("/metrics")
def metrics(_: None = Depends(_require_api_key)) -> PlainTextResponse:
    """Expõe as métricas Prometheus da aplicação.

    Returns:
        Response com métricas no formato Prometheus.
    """
    return PlainTextResponse(generate_latest())


@app.post(f"{API_PREFIX}/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    model: ModelLoader = Depends(get_model),
    normalizer: TextNormalizer = Depends(get_normalizer),
    model_version: str = Depends(get_model_version),
    _: None = Depends(_require_api_key),
) -> PredictResponse:
    """Classifica o nível de urgência de um laudo médico.

    O texto de entrada é normalizado antes da predição para manter
    consistência com o pipeline de treinamento.

    Args:
        request: Requisição com o texto do laudo.
        model: Modelo injetado via dependency injection.
        normalizer: Normalizador injetado via dependency injection.
        model_version: Versão do modelo (prefixo do hash SHA256).
        _: Validação opcional de API key.

    Returns:
        Predição com nível de urgência e confiança.
    """
    start = time.perf_counter()
    normalized_text = normalizer.normalize(request.text)
    if not normalized_text:
        raise HTTPException(
            status_code=422,
            detail="Texto resulta em conteúdo vazio após normalização.",
        )
    urgency, confidence = model.predict(normalized_text)
    prediction_latency_seconds.labels(endpoint="/predict").observe(time.perf_counter() - start)
    predictions_total.labels(urgency=urgency.value).inc()
    return PredictResponse(urgency=urgency, confidence=confidence, model_version=model_version)


@app.post(f"{API_PREFIX}/predict/batch", response_model=list[PredictResponse])
def predict_batch(
    request: PredictBatchRequest,
    model: ModelLoader = Depends(get_model),
    normalizer: TextNormalizer = Depends(get_normalizer),
    model_version: str = Depends(get_model_version),
    _: None = Depends(_require_api_key),
) -> list[PredictResponse]:
    """Classifica um lote de laudos médicos em uma única chamada.

    Args:
        request: Requisição com lista de textos.
        model: Modelo injetado via dependency injection.
        normalizer: Normalizador injetado via dependency injection.
        model_version: Versão do modelo (prefixo do hash SHA256).
        _: Chave de API opcional.

    Returns:
        Lista de predições (urgência + confiança) na ordem de entrada.
    """
    start = time.perf_counter()
    normalized_texts = [normalizer.normalize(text) for text in request.texts]
    if any(not t for t in normalized_texts):
        raise HTTPException(
            status_code=422,
            detail="Um ou mais textos resultam em conteúdo vazio após normalização.",
        )
    results = model.predict_batch(normalized_texts)
    prediction_latency_seconds.labels(endpoint="/predict/batch").observe(
        time.perf_counter() - start
    )
    responses = []
    for urgency, confidence in results:
        predictions_total.labels(urgency=urgency.value).inc()
        responses.append(
            PredictResponse(urgency=urgency, confidence=confidence, model_version=model_version)
        )
    return responses
