"""Pipeline de treinamento do modelo de triagem.

Carrega os dados processados, treina um Pipeline TF-IDF + RandomForest
com seeds fixos e salva o artefato em `models/model.joblib`.
"""

import hashlib
import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.core.dataset import (
    DATA_PROCESSED_PATH,
    EXPECTED_CLASSES,
    MODEL_HASH_PATH,
    MODEL_PATH,
    TEST_DATA_PATH,
    atomic_write_text,
    load_csv_records,
    save_csv_records,
)
from src.core.params import load_params

logger = logging.getLogger(__name__)


def build_pipeline(seed: int, n_estimators: int, max_features: int) -> Pipeline:
    """Constrói o Pipeline TF-IDF + RandomForest.

    Args:
        seed: Seed fixa para reprodutibilidade.
        n_estimators: Número de árvores do RandomForest.
        max_features: Número máximo de features do TF-IDF.

    Returns:
        Pipeline sklearn não treinado.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed)),
        ]
    )


def save_model(model: Pipeline, path: Path) -> None:
    """Persiste o modelo treinado em joblib.

    Args:
        model: Pipeline sklearn treinado.
        path: Caminho do arquivo de saída.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_model_hash(model_path: Path, hash_path: Path) -> None:
    """Calcula e salva o hash SHA256 do modelo para verificação de integridade.

    Args:
        model_path: Caminho do arquivo de modelo.
        hash_path: Caminho do arquivo de hash de saída.
    """
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    atomic_write_text(hash_path, sha256)


def validate_model_classes(model: Pipeline) -> None:
    """Valida que as classes do modelo correspondem ao mapeamento esperado.

    Args:
        model: Pipeline sklearn treinado.

    Raises:
        ValueError: Se as classes não correspondem ao esperado.
    """
    actual = list(model.classes_)
    if actual != EXPECTED_CLASSES:
        raise ValueError(
            f"Classes do modelo {actual} não correspondem ao esperado {EXPECTED_CLASSES}"
        )


def main() -> None:
    """Executa o pipeline de treinamento completo."""
    params = load_params()
    texts, labels = load_csv_records(DATA_PROCESSED_PATH)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=params.train.test_size,
        random_state=params.train.seed,
        stratify=labels,
    )
    pipeline = build_pipeline(
        params.train.seed,
        params.train.random_forest_n_estimators,
        params.train.tfidf_max_features,
    )
    pipeline.fit(train_texts, train_labels)
    validate_model_classes(pipeline)
    save_model(pipeline, MODEL_PATH)
    save_model_hash(MODEL_PATH, MODEL_HASH_PATH)
    save_csv_records(list(zip(test_texts, test_labels)), TEST_DATA_PATH)
    logger.info("Treinamento concluído: modelo salvo em %s", MODEL_PATH)
    logger.info("Split de teste salvo em %s", TEST_DATA_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
