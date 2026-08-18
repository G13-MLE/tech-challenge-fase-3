"""Funções reutilizáveis de treinamento para orquestração (Airflow DAG).

Fornece funções puras (sem dependência do Airflow) que encapsulam as
etapas do pipeline de treinamento: ingestão, carregamento, treino e
persistência. A DAG em dags/train_pipeline.py chama estas funções via
TaskFlow. XCom passa apenas PATHs (strings), nunca objetos.
"""

from pathlib import Path

from src.core.dataset import (
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    MODEL_HASH_PATH,
    MODEL_PATH,
    TEST_DATA_PATH,
    save_csv_records,
)
from src.models.factory import create_model
from src.preprocess.run import build_default_normalizer, preprocess_text
from src.train.run import train_and_save
from src.validate.run import validate_raw_data

__all__ = ["ingest_data", "load_data", "train_model", "save_model", "evaluate_model"]


def ingest_data(raw_path: Path | str | None = None) -> str:
    """Executa o pipeline de ingestão: validação + pré-processamento.

    Args:
        raw_path: Caminho do CSV bruto. Se None, usa o padrão.

    Returns:
        Caminho do CSV processado como string (para XCom).
    """
    if raw_path is None:
        raw_path = DATA_RAW_PATH
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo bruto não encontrado: {raw_path}")
    texts, labels = validate_raw_data(raw_path)
    raw_records = list(zip(texts, labels, strict=True))
    normalizer = build_default_normalizer()
    processed_records = [(preprocess_text(text, normalizer), label) for text, label in raw_records]
    save_csv_records(processed_records, DATA_PROCESSED_PATH)
    return str(DATA_PROCESSED_PATH)


def load_data(path: Path | str | None = None) -> str:
    """Valida e retorna o caminho do CSV processado.

    Args:
        path: Caminho do CSV processado. Se None, usa o padrão.

    Returns:
        Caminho do CSV como string (para XCom).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if path is None:
        path = DATA_PROCESSED_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return str(path)


def train_model(csv_path: str, params=None) -> str:
    """Treina o Pipeline TF-IDF + RandomForest e salva modelo + hash + teste.

    Reutiliza as funções de treino do pipeline DVC (src.train.run).

    Args:
        csv_path: Caminho do CSV processado (recebido via XCom).
        params: Parâmetros do pipeline. Se None, carrega de configs/params.yaml.

    Returns:
        Caminho do modelo salvo como string (para XCom).
    """
    model_path = train_and_save(
        data_path=Path(csv_path),
        model_path=MODEL_PATH,
        model_hash_path=MODEL_HASH_PATH,
        test_data_path=TEST_DATA_PATH,
        params=params,
    )
    return str(model_path)


def save_model(model_path: str) -> str:
    """Valida a integridade do modelo salvo (hash + classes).

    Carrega o modelo via factory pattern, que verifica o hash SHA256
    e valida as classes. Retorna o caminho do modelo.

    Args:
        model_path: Caminho do modelo (recebido via XCom).

    Returns:
        Caminho do modelo validado como string (para XCom).
    """
    create_model(model_path, backend="joblib")
    return model_path


def evaluate_model(model_path: str) -> str:
    """Executa a avaliação do modelo sobre o split de teste.

    Reutiliza o pipeline de avaliação do DVC (src.evaluate.run).

    Args:
        model_path: Caminho do modelo validado (recebido via XCom).

    Returns:
        Caminho do modelo avaliado como string (para XCom).
    """
    from src.evaluate.run import run_evaluation

    run_evaluation(model_path=Path(model_path))
    return model_path
