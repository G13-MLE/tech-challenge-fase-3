"""Testes do pipeline de treinamento."""

import csv

import joblib
import pytest
from sklearn.pipeline import Pipeline

from src.core.dataset import LABEL_COLUMN, TEXT_COLUMN, load_csv_records
from src.core.params import TrainParams
from src.train.run import (
    build_pipeline,
    resolve_stopwords,
    save_feature_importances,
    save_model,
    validate_model_classes,
)

SEED = 42
N_ESTIMATORS = 10
MAX_FEATURES = 100

TEST_PARAMS = TrainParams(
    seed=SEED,
    test_size=0.2,
    random_forest_n_estimators=N_ESTIMATORS,
    tfidf_max_features=MAX_FEATURES,
)

# Corpus com termos repetidos para respeitar min_df=2 em datasets pequenos.
FIT_TEXTS = [
    "paciente com asma",
    "paciente com asma",
    "pneumonia grave",
    "pneumonia grave",
    "diabetes tipo 2",
    "diabetes tipo 2",
]
FIT_LABELS = [0, 0, 2, 2, 1, 1]


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
        pipeline = build_pipeline(TEST_PARAMS)
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["tfidf", "clf"]

    def test_build_pipeline_tfidf_configuration(self) -> None:
        params = TEST_PARAMS.model_copy(update={"tfidf_stopwords": True})
        pipeline = build_pipeline(params)
        tfidf = pipeline.named_steps["tfidf"]
        assert tfidf.ngram_range == (1, 2)
        assert tfidf.sublinear_tf is True
        assert tfidf.min_df == 2
        assert tfidf.max_df == 0.95
        assert tfidf.max_features == MAX_FEATURES
        stopwords = tfidf.stop_words
        assert stopwords is not None
        assert "the" in stopwords
        assert "and" in stopwords

    def test_build_pipeline_class_weight_balanced(self) -> None:
        params = TEST_PARAMS.model_copy(update={"tfidf_stopwords": True})
        pipeline = build_pipeline(params)
        assert pipeline.named_steps["clf"].class_weight == "balanced"

    def test_build_pipeline_without_stopwords(self) -> None:
        params = TEST_PARAMS.model_copy(update={"tfidf_stopwords": False})
        pipeline = build_pipeline(params)
        assert pipeline.named_steps["tfidf"].stop_words is None

    def test_build_pipeline_custom_stopwords(self) -> None:
        params = TEST_PARAMS.model_copy(update={"tfidf_stopwords": ["custom"]})
        pipeline = build_pipeline(params)
        assert pipeline.named_steps["tfidf"].stop_words == ["custom"]

    def test_build_pipeline_is_deterministic(self) -> None:
        first = build_pipeline(TEST_PARAMS)
        second = build_pipeline(TEST_PARAMS)
        first.fit(FIT_TEXTS, FIT_LABELS)
        second.fit(FIT_TEXTS, FIT_LABELS)
        assert first.predict(["paciente com pneumonia"]) == second.predict(
            ["paciente com pneumonia"]
        )


class TestResolveStopwords:
    """Testes da resolução de stopwords."""

    def test_true_returns_english_list(self) -> None:
        params = TEST_PARAMS.model_copy(update={"tfidf_stopwords": True})
        stopwords = resolve_stopwords(params)
        assert stopwords is not None
        assert "the" in stopwords

    def test_false_returns_none(self) -> None:
        params = TEST_PARAMS.model_copy(update={"tfidf_stopwords": False})
        assert resolve_stopwords(params) is None

    def test_list_returns_as_is(self) -> None:
        params = TEST_PARAMS.model_copy(update={"tfidf_stopwords": ["x", "y"]})
        assert resolve_stopwords(params) == ["x", "y"]


class TestSaveModel:
    """Testes da persistência do modelo."""

    def test_save_model_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "model.joblib"
        pipeline = build_pipeline(TEST_PARAMS)
        pipeline.fit(FIT_TEXTS, FIT_LABELS)
        save_model(pipeline, path)
        loaded = joblib.load(path)
        assert loaded.predict(["pneumonia grave"]) == [2]


class TestValidateModelClasses:
    """Testes da validação de classes do modelo."""

    def test_validate_correct_classes(self) -> None:
        pipeline = build_pipeline(TEST_PARAMS)
        pipeline.fit(FIT_TEXTS, FIT_LABELS)
        validate_model_classes(pipeline)

    def test_validate_incomplete_classes_raises(self) -> None:
        pipeline = build_pipeline(TEST_PARAMS)
        # 12 documentos com pares de termos; cada token aparece em 6 docs
        # (dentro de min_df=2 / max_df=0.95 para não ser podado).
        pairs = [
            ("asma bronquica", "sibilos"),
            ("asma bronquica", "dispneia"),
            ("asma bronquica", "tosse"),
            ("sibilos", "dispneia"),
            ("sibilos", "tosse"),
            ("dispneia", "tosse"),
        ]
        texts = [f"{a} {b}" for a, b in pairs] * 2
        labels = [0] * len(texts)
        pipeline.fit(texts, labels)
        with pytest.raises(ValueError, match="não correspondem"):
            validate_model_classes(pipeline)


class TestFeatureImportances:
    """Testes da extração de importâncias de features."""

    def test_save_feature_importances(self, tmp_path) -> None:
        path = tmp_path / "feature_importances.csv"
        pipeline = build_pipeline(TEST_PARAMS)
        pipeline.fit(FIT_TEXTS, FIT_LABELS)
        save_feature_importances(pipeline, path)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("feature,importance\n")
        assert "pneumonia" in content
