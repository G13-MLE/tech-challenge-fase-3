"""Configurações centralizadas do projeto via Pydantic Settings.

Carrega variáveis de ambiente do arquivo `.env` (quando existente) e
disponibiliza acesso tipado às configurações em toda a aplicação.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação.

    Attributes:
        app_name: Nome da aplicação.
        app_env: Ambiente de execução (development | production).
        log_level: Nível de log (DEBUG | INFO | WARNING | ERROR).
        api_port: Porta do servidor FastAPI.
        model_path: Caminho do artefato de modelo treinado.
        mlflow_tracking_uri: URL do servidor MLflow.
        kaggle_username: Usuário do Kaggle para download do dataset.
        kaggle_key: Chave de API do Kaggle.
        dvc_onedrive_remote_url: URL do remote DVC (OneDrive), opcional.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "triage-api"
    app_env: str = "development"
    log_level: str = "INFO"
    api_port: int = 8000
    model_path: Path = Path("models/model.joblib")
    mlflow_tracking_uri: str = "http://localhost:5001"
    kaggle_username: str = ""
    kaggle_key: str = ""
    dvc_onedrive_remote_url: str = ""


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância única (cacheada) das configurações.

    Returns:
        Settings: Configurações carregadas do ambiente.
    """
    return Settings()
