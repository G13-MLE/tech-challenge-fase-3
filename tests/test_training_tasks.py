"""Testes do módulo de orquestração de treinamento."""

import csv as csv_mod
from pathlib import Path

import pytest

from src.core.params import TrainParams
from src.orchestration.training_tasks import (
    evaluate_model,
    load_data,
    save_model,
    train_model,
)

TRAINING_ROWS = [
    ("paciente com asma brônquica e sibilos", 0),
    ("asma crônica com sibilos", 0),
    ("crise asmática com broncoespasmo", 0),
    ("hérnia inguinal sem complicação", 0),
    ("hérnia umbilical redutível", 0),
    ("diabetes tipo 2 descompensada", 1),
    ("diabetes com hemoglobina glicada alta", 1),
    ("hipertensão arterial crônica", 1),
    ("crise hipertensiva com cefaleia", 1),
    ("hipertensão não controlada", 1),
    ("pneumonia bacteriana grave", 2),
    ("pneumonia lobar com febre alta", 2),
    ("pneumonia comunitária com infiltrado", 2),
    ("pneumonia viral com dispneia", 2),
    ("pneumonia nosocomial em UTI", 2),
]

TEST_PARAMS = TrainParams(
    seed=42, test_size=0.2, random_forest_n_estimators=10, tfidf_max_features=100
)


def write_csv(path: Path) -> Path:
    """Escreve CSV de treino com dados suficientes para stratified split."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(TRAINING_ROWS)
    return path


class TestLoadData:
    """Testes de load_data."""

    def test_load_data_returns_path(self, tmp_path):
        csv = tmp_path / "processed.csv"
        csv.write_text("text,label\npaciente com asma,0\n", encoding="utf-8")
        result = load_data(csv)
        assert result == str(csv)

    def test_load_data_default_path_exists(self, tmp_path, monkeypatch):
        csv = tmp_path / "processed.csv"
        csv.write_text("text,label\npaciente com asma,0\n", encoding="utf-8")
        monkeypatch.setattr("src.orchestration.training_tasks.DATA_PROCESSED_PATH", csv)
        result = load_data()
        assert result == str(csv)

    def test_load_data_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "inexistente.csv")


class TestTrainModel:
    """Testes de train_model."""

    def test_train_model_saves_artifacts(self, tmp_path, monkeypatch):
        csv_path = write_csv(tmp_path / "processed.csv")

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_path = model_dir / "model.joblib"
        hash_path = model_dir / "model.joblib.sha256"
        test_path = tmp_path / "test_split.csv"
        feat_path = tmp_path / "feature_importances.csv"
        train_metrics_path = tmp_path / "train_metrics.json"

        monkeypatch.setattr("src.orchestration.training_tasks.MODEL_PATH", model_path)
        monkeypatch.setattr("src.orchestration.training_tasks.MODEL_HASH_PATH", hash_path)
        monkeypatch.setattr("src.orchestration.training_tasks.TEST_DATA_PATH", test_path)
        monkeypatch.setattr("src.train.run.FEATURE_IMPORTANCES_PATH", feat_path)
        monkeypatch.setattr("src.train.run.TRAIN_METRICS_PATH", train_metrics_path)

        result = train_model(str(csv_path), params=TEST_PARAMS)

        assert result == str(model_path)
        assert model_path.exists()
        assert hash_path.exists()
        assert test_path.exists()

    def test_train_model_deterministic(self, tmp_path, monkeypatch):
        csv_path = write_csv(tmp_path / "processed.csv")

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_path = model_dir / "model.joblib"
        hash_path = model_dir / "model.joblib.sha256"
        test_path = tmp_path / "test_split.csv"

        monkeypatch.setattr("src.orchestration.training_tasks.MODEL_PATH", model_path)
        monkeypatch.setattr("src.orchestration.training_tasks.MODEL_HASH_PATH", hash_path)
        monkeypatch.setattr("src.orchestration.training_tasks.TEST_DATA_PATH", test_path)
        monkeypatch.setattr(
            "src.train.run.FEATURE_IMPORTANCES_PATH", tmp_path / "feature_importances.csv"
        )
        monkeypatch.setattr("src.train.run.TRAIN_METRICS_PATH", tmp_path / "train_metrics.json")

        train_model(str(csv_path), params=TEST_PARAMS)
        hash1 = hash_path.read_text(encoding="utf-8").strip()
        first_bytes = model_path.read_bytes()

        train_model(str(csv_path), params=TEST_PARAMS)
        hash2 = hash_path.read_text(encoding="utf-8").strip()
        second_bytes = model_path.read_bytes()

        assert hash1 == hash2
        assert first_bytes == second_bytes


class TestSaveModel:
    """Testes de save_model (validação de integridade)."""

    def test_save_model_validates(self, mini_model_path):
        result = save_model(str(mini_model_path))
        assert result == str(mini_model_path)

    def test_save_model_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            save_model(str(tmp_path / "inexistente.joblib"))


class TestEvaluateModel:
    """Testes de evaluate_model."""

    def test_evaluate_model_runs(self, mini_model_path, tmp_path, monkeypatch):
        test_data = tmp_path / "test_split.csv"
        test_data.write_text(
            "text,label\npaciente com asma,0\npneumonia grave,2\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.evaluate.run.TEST_DATA_PATH", test_data)
        monkeypatch.setattr("src.evaluate.run.METRICS_PATH", tmp_path / "metrics.json")
        monkeypatch.setattr("src.evaluate.run.REPORT_PATH", tmp_path / "classification_report.txt")
        monkeypatch.setattr(
            "src.evaluate.run.CONFUSION_MATRIX_PATH", tmp_path / "confusion_matrix.csv"
        )
        result = evaluate_model(str(mini_model_path))
        assert result == str(mini_model_path)
