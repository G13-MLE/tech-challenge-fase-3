"""Utilitários opcionais de integração com MLflow.

O MLflow é usado para registrar parâmetros, métricas e artefatos dos
estágios de treino e avaliação. A integração é GRACEFUL: se o servidor
MLflow não estiver disponível (ou o pacote não estiver instalado),
apenas um warning é emitido e o pipeline continua normalmente.
"""

import logging
from pathlib import Path
from typing import Any

from src.core.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["mlflow_log_run", "mlflow_log_metrics", "mlflow_log_params", "mlflow_log_artifact"]


def _get_mlflow(settings: Settings):
    """Retorna o módulo mlflow ou None se indisponível.

    Args:
        settings: Configurações da aplicação.

    Returns:
        Módulo mlflow ou None se não instalado.
    """
    if not settings.mlflow_tracking_uri:
        return None
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("Pacote mlflow não instalado — logging MLflow desativado.")
        return None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow


def mlflow_log_params(settings: Settings, params: dict[str, Any]) -> None:
    """Registra parâmetros no MLflow de forma opcional.

    Args:
        settings: Configurações da aplicação.
        params: Dicionário de parâmetros.
    """
    mlflow = _get_mlflow(settings)
    if mlflow is None:
        return
    try:
        mlflow.log_params(params)
    except Exception:
        logger.warning("Falha ao registrar parâmetros no MLflow (ignorado).", exc_info=True)


def mlflow_log_metrics(settings: Settings, metrics: dict[str, float]) -> None:
    """Registra métricas no MLflow de forma opcional.

    Args:
        settings: Configurações da aplicação.
        metrics: Dicionário de métricas numéricas.
    """
    mlflow = _get_mlflow(settings)
    if mlflow is None:
        return
    try:
        mlflow.log_metrics(metrics)
    except Exception:
        logger.warning("Falha ao registrar métricas no MLflow (ignorado).", exc_info=True)


def mlflow_log_artifact(settings: Settings, path: Path) -> None:
    """Registra um artefato (arquivo/diretório) no MLflow de forma opcional.

    Args:
        settings: Configurações da aplicação.
        path: Caminho do artefato a registrar.
    """
    mlflow = _get_mlflow(settings)
    if mlflow is None:
        return
    if not path.exists():
        logger.warning("Artefato não encontrado, ignorando MLflow: %s", path)
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception:
        logger.warning("Falha ao registrar artefato no MLflow (ignorado).", exc_info=True)


def mlflow_log_run(
    settings: Settings,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
) -> None:
    """Executa um run MLflow com parâmetros, métricas e artefatos.

    Se o MLflow estiver indisponível, emite warning e retorna sem erro.

    Args:
        settings: Configurações da aplicação.
        run_name: Nome do run (ex.: "treino", "avaliacao").
        params: Dicionário de parâmetros.
        metrics: Dicionário de métricas numéricas.
        artifacts: Lista de caminhos de artefatos a registrar.
    """
    mlflow = _get_mlflow(settings)
    if mlflow is None:
        logger.info("MLflow indisponível — run '%s' ignorado.", run_name)
        return
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for artifact in artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
    except Exception:
        logger.warning("Falha no run MLflow '%s' (ignorado).", run_name, exc_info=True)
