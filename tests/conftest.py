"""Fixtures compartilhadas entre os testes."""

import csv

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.core.dataset import LABEL_COLUMN, TEXT_COLUMN

SEED = 42
N_ESTIMATORS = 10
MAX_FEATURES = 100

TRAINING_DATA = [
    ("paciente com asma brônquica e sibilos", 0),
    ("asma crônica com sibilos", 0),
    ("crise asmática com broncoespasmo", 0),
    ("hérnia inguinal sem complicação", 0),
    ("hérnia umbilical redutível", 0),
    ("diabetes tipo 2 descompensada", 1),
    ("diabetes com hemoglobina glicada alta", 1),
    ("hipertensão arterial crônica", 1),
    ("crise hipertensiva com cefaleia", 1),
    ("pneumonia bacteriana grave", 2),
    ("pneumonia lobar com febre alta", 2),
    ("pneumonia comunitária com infiltrado", 2),
]


def build_mini_model(path) -> Pipeline:
    """Treina e salva um modelo mínimo em tmp_path para os testes."""
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES)),
            ("clf", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, path)
    return pipeline


def write_raw_csv(path, rows: list[tuple[str, int]]) -> None:
    """Escreve um CSV bruto de laudos para os testes."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([TEXT_COLUMN, LABEL_COLUMN])
        writer.writerows(rows)


@pytest.fixture()
def mini_model_path(tmp_path):
    """Caminho para um modelo treinado em tmp_path."""
    model_path = tmp_path / "model.joblib"
    build_mini_model(model_path)
    return model_path


@pytest.fixture()
def mini_model(mini_model_path):
    """Modelo treinado em tmp_path."""
    from src.models.factory import JoblibLoader

    loader = JoblibLoader()
    loader.load(mini_model_path)
    return loader
