"""Testes do pipeline de treinamento."""

import csv

import joblib
import pytest
from sklearn.pipeline import Pipeline

from src.core.dataset import LABEL_COLUMN, TEXT_COLUMN, load_csv_records
from src.train.run import build_pipeline, save_model, validate_model_classes

SEED = 42
N_ESTIMATORS = 10
MAX_FEATURES = 100


def write_processed_csv(path, rows: list[tuple[str, int]]) -> None:
    """Escreve um CSV processado de laudos para os testes."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([TEXT_COLUMN, LABEL_COLUMN])
        writer.writerows(rows)


class TestLoadProcessedRecords:
    """Testes de carregamento dos dados processados."""

    def test_load_csv_records(self, tmp_path) -> None:
        path = tmp_path / "processed.csv"
        write_processed_csv(path, [("paciente com asma", 0), ("pneumonia grave", 2)])
        texts, labels = load_csv_records(path)
        assert texts == ["paciente com asma", "pneumonia grave"]
        assert labels == [0, 2]

    def test_load_csv_records_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv_records(tmp_path / "inexistente.csv")


class TestBuildPipeline:
    """Testes da construção do Pipeline."""

    def test_build_pipeline_steps(self) -> None:
        pipeline = build_pipeline(SEED, N_ESTIMATORS, MAX_FEATURES)
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["tfidf", "clf"]

    def test_build_pipeline_is_deterministic(self) -> None:
        first = build_pipeline(SEED, N_ESTIMATORS, MAX_FEATURES)
        second = build_pipeline(SEED, N_ESTIMATORS, MAX_FEATURES)
        texts = ["paciente com asma", "pneumonia grave", "diabetes tipo 2"]
        labels = [0, 2, 1]
        first.fit(texts, labels)
        second.fit(texts, labels)
        assert first.predict(["paciente com pneumonia"]) == second.predict(
            ["paciente com pneumonia"]
        )


class TestSaveModel:
    """Testes da persistência do modelo."""

    def test_save_model_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "model.joblib"
        pipeline = build_pipeline(SEED, N_ESTIMATORS, MAX_FEATURES)
        pipeline.fit(["paciente com asma", "pneumonia grave"], [0, 2])
        save_model(pipeline, path)
        loaded = joblib.load(path)
        assert loaded.predict(["pneumonia grave"]) == [2]


class TestValidateModelClasses:
    """Testes da validação de classes do modelo."""

    def test_validate_correct_classes(self) -> None:
        pipeline = build_pipeline(SEED, N_ESTIMATORS, MAX_FEATURES)
        pipeline.fit(["asma", "diabetes", "pneumonia"], [0, 1, 2])
        validate_model_classes(pipeline)

    def test_validate_incomplete_classes_raises(self) -> None:
        pipeline = build_pipeline(SEED, N_ESTIMATORS, MAX_FEATURES)
        pipeline.fit(["asma", "asma"], [0, 0])
        with pytest.raises(ValueError, match="não correspondem"):
            validate_model_classes(pipeline)
