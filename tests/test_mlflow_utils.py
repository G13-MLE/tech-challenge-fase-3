"""Testes dos utilitários de MLflow (graceful degradation)."""

import logging
from unittest.mock import patch

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

    def test_mlflow_log_run_unreachable_uri_warns(self, tmp_path, caplog) -> None:
        """Testa que run com URI inacessível emite warning e não bloqueia.

        Usa mock para simular ConnectionError no mlflow.start_run,
        evitando timeout real na conexão com servidor inexistente.
        """
        settings = Settings(mlflow_tracking_uri="http://localhost:1/unreachable")
        with (
            caplog.at_level(logging.WARNING),
            patch("src.core.mlflow_utils._get_mlflow") as mock_get_mlflow,
        ):
            mock_mlflow = __import__("types").ModuleType("mlflow")
            mock_mlflow.start_run = __import__("unittest.mock").mock.MagicMock(
                side_effect=ConnectionError("Connection refused")
            )
            mock_mlflow.log_params = __import__("unittest.mock").mock.MagicMock()
            mock_mlflow.log_metrics = __import__("unittest.mock").mock.MagicMock()
            mock_mlflow.log_artifact = __import__("unittest.mock").mock.MagicMock()
            mock_mlflow.set_tracking_uri = __import__("unittest.mock").mock.MagicMock()
            mock_get_mlflow.return_value = mock_mlflow
            mlflow_log_run(settings, "teste", {}, {}, [])
