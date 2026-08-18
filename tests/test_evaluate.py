"""Testes do pipeline de avaliação."""

import json

import pytest

from src.core.dataset import load_csv_records
from src.evaluate.run import (
    compute_metrics,
    save_confusion_matrix,
    save_metrics,
    save_report,
)

METRICS_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "precision_normal",
    "recall_normal",
    "f1_normal",
    "precision_atencao",
    "recall_atencao",
    "f1_atencao",
    "precision_urgente",
    "recall_urgente",
    "f1_urgente",
}


class TestLoadEvaluationData:
    """Testes de carregamento dos dados para avaliação."""

    def test_load_csv_records(self, tmp_path) -> None:
        path = tmp_path / "processed.csv"
        path.write_text("text,label\npaciente com asma,0\npneumonia grave,2\n", encoding="utf-8")
        texts, labels = load_csv_records(path)
        assert texts == ["paciente com asma", "pneumonia grave"]
        assert labels == [0, 2]

    def test_load_csv_records_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv_records(tmp_path / "inexistente.csv")


class TestComputeMetrics:
    """Testes do cálculo de métricas."""

    def test_metrics_perfect_predictions(self) -> None:
        metrics = compute_metrics([0, 1, 2], [0, 1, 2])
        assert set(metrics) == METRICS_KEYS
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["recall_urgente"] == 1.0

    def test_metrics_mixed_predictions(self) -> None:
        metrics = compute_metrics([0, 0, 1, 1, 2], [0, 0, 1, 2, 2])
        assert set(metrics) == METRICS_KEYS
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["recall_urgente"] <= 1.0

    def test_metrics_all_wrong(self) -> None:
        metrics = compute_metrics([0, 1, 2], [1, 2, 0])
        assert metrics["accuracy"] == 0.0
        assert metrics["recall_urgente"] == 0.0

    def test_recall_urgente_specific(self) -> None:
        # 2 urgente real: apenas 1 dos 2 detectado
        metrics = compute_metrics([0, 2, 2], [0, 0, 2])
        assert metrics["recall_urgente"] == 0.5


class TestSaveArtifacts:
    """Testes da persistência de métricas, relatório e matriz."""

    def test_save_metrics_json(self, tmp_path) -> None:
        path = tmp_path / "metrics.json"
        metrics = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "recall_urgente": 1.0,
        }
        save_metrics(metrics, path)
        with path.open(encoding="utf-8") as file:
            assert json.load(file) == metrics

    def test_save_report_text(self, tmp_path) -> None:
        path = tmp_path / "report.txt"
        save_report([0, 1, 2], [0, 1, 2], path)
        content = path.read_text(encoding="utf-8")
        assert "precision" in content
        assert "recall" in content
        assert "f1-score" in content
        assert "urgente" in content

    def test_save_confusion_matrix_csv(self, tmp_path) -> None:
        path = tmp_path / "confusion_matrix.csv"
        save_confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], path)
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert lines[0] == "true_label,pred_normal,pred_atencao,pred_urgente"
        assert len(lines) == 4
        assert lines[1].startswith("normal,")
        assert lines[3].startswith("urgente,")
