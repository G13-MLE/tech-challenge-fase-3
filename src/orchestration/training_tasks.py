"""Funções reutilizáveis de treinamento para orquestração (Airflow DAG).

Fornece funções puras (sem dependência do Airflow) que encapsulam as
etapas do pipeline de treinamento: carregamento, treino e persistência.
A DAG em dags/train_pipeline.py chama estas funções via TaskFlow.
"""

from pathlib import Path

from src.core.dataset import (
    DATA_PROCESSED_PATH,
    MODEL_HASH_PATH,
    MODEL_PATH,
    TEST_DATA_PATH,
)
from src.models.factory import create_model
from src.train.run import train_and_save

__all__ = ["load_data", "train_model", "save_model"]


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
