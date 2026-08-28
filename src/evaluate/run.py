"""Avaliação do modelo de triagem.

Carrega o modelo treinado e os dados de teste, calcula métricas de
classificação (macro + por classe) e salva os resultados em `reports`.
"""

import json
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.core.dataset import (
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    MODEL_PATH,
    REPORT_PATH,
    TEST_DATA_PATH,
    atomic_write_text,
    load_csv_records,
)
from src.models.factory import ModelBackend, create_model
from src.models.urgency import URGENCY_MAP

logger = logging.getLogger(__name__)

__all__ = [
    "CLASS_NAMES",
    "compute_metrics",
    "save_metrics",
    "save_report",
    "save_confusion_matrix",
    "run_evaluation",
    "main",
]

CLASS_NAMES = ["normal", "atencao", "urgente"]  # noqa: E501 — ASCII for CSV headers; API uses "atenção"


def compute_metrics(true_labels: list[int], predicted_labels: list[int]) -> dict[str, float]:
    """Calcula as métricas de classificação do modelo.

    Inclui métricas macro e per-class (precision, recall, F1), além de
    `recall_urgente` (recall da classe 2, a mais crítica para triagem).

    Args:
        true_labels: Rótulos reais.
        predicted_labels: Rótulos preditos.

    Returns:
        Dicionário com acurácia, precision/recall/F1 macro e per-class.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="macro"),
        "recall": recall_score(true_labels, predicted_labels, average="macro"),
        "f1": f1_score(true_labels, predicted_labels, average="macro"),
    }
    labels = [0, 1, 2]
    per_class_precision = precision_score(
        true_labels, predicted_labels, average=None, labels=labels, zero_division=0
    )
    per_class_recall = recall_score(
        true_labels, predicted_labels, average=None, labels=labels, zero_division=0
    )
    per_class_f1 = f1_score(
        true_labels, predicted_labels, average=None, labels=labels, zero_division=0
    )
    for index, name in enumerate(CLASS_NAMES):
        metrics[f"precision_{name}"] = per_class_precision[index]
        metrics[f"recall_{name}"] = per_class_recall[index]
        metrics[f"f1_{name}"] = per_class_f1[index]
    return metrics


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
    content = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    atomic_write_text(path, content)


def save_confusion_matrix(true_labels: list[int], predicted_labels: list[int], path: Path) -> None:
    """Persiste a matriz de confusão em CSV de forma atômica.

    Args:
        true_labels: Rótulos reais.
        predicted_labels: Rótulos preditos.
        path: Caminho do arquivo CSV de saída.
    """
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
    header = "true_label," + ",".join(f"pred_{name}" for name in CLASS_NAMES)
    rows = [
        f"{name}," + ",".join(str(value) for value in row)
        for name, row in zip(CLASS_NAMES, matrix, strict=True)
    ]
    atomic_write_text(path, header + "\n" + "\n".join(rows) + "\n")


def run_evaluation(
    model_path: Path = MODEL_PATH,
    test_data_path: Path | None = None,
    metrics_path: Path = METRICS_PATH,
    report_path: Path = REPORT_PATH,
    confusion_matrix_path: Path = CONFUSION_MATRIX_PATH,
    backend: ModelBackend = "joblib",
) -> dict[str, float]:
    """Executa a avaliação completa e salva métricas, relatório e matriz.

    Args:
        model_path: Caminho do modelo a avaliar.
        test_data_path: Caminho do CSV de teste. Se None, usa o padrão.
        metrics_path: Caminho do JSON de métricas de saída.
        report_path: Caminho do relatório de classificação de saída.
        confusion_matrix_path: Caminho do CSV da matriz de confusão de saída.
        backend: Backend de carregamento ('joblib' ou 'onnx').

    Returns:
        Dicionário com as métricas calculadas.
    """
    if test_data_path is None:
        test_data_path = TEST_DATA_PATH
    texts, labels = load_csv_records(test_data_path)
    model = create_model(model_path, backend=backend)
    predictions = model.predict_batch(texts)
    label_to_int = {v: k for k, v in URGENCY_MAP.items()}
    predicted_labels = [label_to_int[u] for u, _ in predictions]
    metrics = compute_metrics(labels, predicted_labels)
    save_metrics(metrics, metrics_path)
    save_report(labels, predicted_labels, report_path)
    save_confusion_matrix(labels, predicted_labels, confusion_matrix_path)
    logger.info("Avaliação concluída: métricas salvas em %s", metrics_path)
    logger.info(
        "Acurácia: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | Recall urgente: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["recall_urgente"],
    )
    return metrics


def main() -> None:
    """Executa a avaliação do modelo e salva métricas, relatório e matriz."""
    run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
