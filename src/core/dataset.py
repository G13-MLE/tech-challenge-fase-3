"""Caminhos e utilitários de persistência do pipeline.

Centraliza os caminhos de dados, modelo e relatórios usados pelos
estágios do DVC, evitando literais espalhados pelo código.
"""

import csv
from pathlib import Path

__all__ = [
    "DATA_RAW_PATH",
    "DATA_PROCESSED_PATH",
    "TEST_DATA_PATH",
    "MODEL_PATH",
    "MODEL_HASH_PATH",
    "ONNX_MODEL_PATH",
    "ONNX_MODEL_HASH_PATH",
    "THRESHOLDS_PATH",
    "METRICS_PATH",
    "REPORT_PATH",
    "CONFUSION_MATRIX_PATH",
    "FEATURE_IMPORTANCES_PATH",
    "TRAIN_METRICS_PATH",
    "TEXT_COLUMN",
    "LABEL_COLUMN",
    "EXPECTED_CLASSES",
    "load_csv_records",
    "save_csv_records",
    "atomic_write_text",
    "atomic_write_bytes",
]

DATA_RAW_PATH = Path("data/raw/laudos.csv")
DATA_PROCESSED_PATH = Path("data/processed/laudos_processed.csv")
TEST_DATA_PATH = Path("data/processed/test_split.csv")
MODEL_PATH = Path("models/model.joblib")
MODEL_HASH_PATH = Path("models/model.joblib.sha256")
ONNX_MODEL_PATH = Path("models/model.onnx")
ONNX_MODEL_HASH_PATH = Path("models/model.onnx.sha256")
THRESHOLDS_PATH = Path("models/thresholds.json")
METRICS_PATH = Path("reports/metrics.json")
REPORT_PATH = Path("reports/classification_report.txt")
CONFUSION_MATRIX_PATH = Path("reports/confusion_matrix.csv")
FEATURE_IMPORTANCES_PATH = Path("reports/feature_importances.csv")
TRAIN_METRICS_PATH = Path("reports/train_metrics.json")

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

EXPECTED_CLASSES = [0, 1, 2]


def load_csv_records(path: Path) -> tuple[list[str], list[int]]:
    """Carrega textos e rótulos de um CSV processado.

    Args:
        path: Caminho do CSV com as colunas text e label.

    Returns:
        Tupla (textos, rótulos).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    texts: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row[TEXT_COLUMN])
            labels.append(int(row[LABEL_COLUMN]))
    return texts, labels


def save_csv_records(records: list[tuple[str, int]], path: Path) -> None:
    """Persiste registros em CSV de forma atômica.

    Escreve em arquivo temporário e renomeia, evitando arquivos
    parciais em caso de falha.

    Args:
        records: Lista de tuplas (texto, rótulo).
        path: Caminho do arquivo CSV de saída.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([TEXT_COLUMN, LABEL_COLUMN])
            writer.writerows(records)
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Escreve conteúdo textual de forma atômica.

    Args:
        path: Caminho do arquivo de saída.
        content: Conteúdo a ser escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Escreve conteúdo binário de forma atômica.

    Args:
        path: Caminho do arquivo de saída.
        content: Conteúdo binário a ser escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_bytes(content)
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
