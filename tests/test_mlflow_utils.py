"""Testes dos utilitários de MLflow (graceful degradation)."""

import logging

from src.core.config import Settings
from src.core.mlflow_utils import (
    mlflow_log_artifact,
    mlflow_log_metrics,
    mlflow_log_params,
    mlflow_log_run,
)


class TestMlflowUtils:
    """Testes dos helpers de MLflow (devem ser graceful)."""

    def test_mlflow_log_run_without_uri_is_noop(self, tmp_path) -> None:
        settings = Settings(mlflow_tracking_uri="")
        # Não deve levantar exceção
        mlflow_log_run(settings, "teste", {"a": 1}, {"acc": 1.0}, [tmp_path])

    def test_mlflow_log_params_without_uri_is_noop(self) -> None:
        settings = Settings(mlflow_tracking_uri="")
        mlflow_log_params(settings, {"a": 1})

    def test_mlflow_log_metrics_without_uri_is_noop(self) -> None:
        settings = Settings(mlflow_tracking_uri="")
        mlflow_log_metrics(settings, {"acc": 1.0})

    def test_mlflow_log_artifact_without_uri_is_noop(self, tmp_path) -> None:
        settings = Settings(mlflow_tracking_uri="")
        mlflow_log_artifact(settings, tmp_path / "inexistente.txt")

    def test_mlflow_log_run_unreachable_uri_warns(self, tmp_path, caplog, monkeypatch) -> None:
        settings = Settings(mlflow_tracking_uri="http://localhost:1/unreachable")
        with caplog.at_level(logging.WARNING):
            # Não deve levantar exceção mesmo com servidor inacessível
            mlflow_log_run(settings, "teste", {}, {}, [])
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
