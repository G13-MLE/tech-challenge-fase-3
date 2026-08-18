"""Avaliação do modelo de triagem.

Carrega o modelo treinado e os dados de teste, calcula métricas de
classificação e salva os resultados em `reports`.
"""

import json
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.core.dataset import (
    METRICS_PATH,
    MODEL_PATH,
    REPORT_PATH,
    TEST_DATA_PATH,
    atomic_write_text,
    load_csv_records,
)
from src.models.factory import create_model
from src.models.urgency import URGENCY_MAP

logger = logging.getLogger(__name__)


def compute_metrics(true_labels: list[int], predicted_labels: list[int]) -> dict[str, float]:
    """Calcula as métricas de classificação do modelo.

    Args:
        true_labels: Rótulos reais.
        predicted_labels: Rótulos preditos.

    Returns:
        Dicionário com acurácia, precision, recall e F1 (macro).
    """
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="macro"),
        "recall": recall_score(true_labels, predicted_labels, average="macro"),
        "f1": f1_score(true_labels, predicted_labels, average="macro"),
    }


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    """Persiste as métricas em JSON de forma atômica.

    Args:
        metrics: Dicionário de métricas calculadas.
        path: Caminho do arquivo JSON de saída.
    """
    content = json.dumps(metrics, indent=2) + "\n"
    atomic_write_text(path, content)


def save_report(true_labels: list[int], predicted_labels: list[int], path: Path) -> None:
    """Persiste o relatório de classificação em texto de forma atômica.

    Args:
        true_labels: Rótulos reais.
        predicted_labels: Rótulos preditos.
        path: Caminho do arquivo de saída.
    """
    content = classification_report(true_labels, predicted_labels, zero_division=0)
    atomic_write_text(path, content)


def main() -> None:
    """Executa a avaliação do modelo e salva métricas e relatório."""
    texts, labels = load_csv_records(TEST_DATA_PATH)
    model = create_model(MODEL_PATH, backend="joblib")
    predictions = [model.predict(text) for text in texts]
    label_to_int = {v: k for k, v in URGENCY_MAP.items()}
    predicted_labels = [label_to_int[u] for u, _ in predictions]
    metrics = compute_metrics(labels, predicted_labels)
    save_metrics(metrics, METRICS_PATH)
    save_report(labels, predicted_labels, REPORT_PATH)
    logger.info("Avaliação concluída: métricas salvas em %s", METRICS_PATH)
    logger.info(
        "Acurácia: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
