"""Configurações centralizadas do projeto via Pydantic Settings.

Carrega variáveis de ambiente do arquivo `.env` (quando existente) e
disponibiliza acesso tipado às configurações em toda a aplicação.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Configurações da aplicação.

    Attributes:
        app_name: Nome da aplicação.
        app_env: Ambiente de execução (development | production).
        log_level: Nível de log (DEBUG | INFO | WARNING | ERROR).
        api_port: Porta do servidor FastAPI.
        model_path: Caminho do artefato de modelo treinado.
        model_backend: Backend de carregamento (joblib | onnx).
        mlflow_tracking_uri: URL do servidor MLflow.
        kaggle_username: Usuário do Kaggle para download do dataset.
        kaggle_key: Chave de API do Kaggle (armazenada como segredo).
        dvc_onedrive_remote_url: URL do remote DVC (OneDrive), opcional.
        cors_allow_origins: Lista de origens permitidas para CORS
            ('*' permite todas as origens).
        api_key_enabled: Se True, exige header X-API-Key em /predict.
        api_key: Chave de API para autenticação (armazenada como segredo).
        rate_limit_requests: Número máximo de requisições por IP por janela.
        rate_limit_window_seconds: Janela do rate limit em segundos.
        trust_forwarded_headers: Se True, confia no header X-Forwarded-For
            para extrair o IP do cliente (use apenas atrás de proxy reverso).
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
    model_backend: Literal["joblib", "onnx"] = "joblib"
    mlflow_tracking_uri: str = "http://localhost:5001"
    kaggle_username: str = ""
    kaggle_key: SecretStr = SecretStr("")
    dvc_onedrive_remote_url: str = ""
    cors_allow_origins: list[str] = ["*"]
    api_key_enabled: bool = False
    api_key: SecretStr = SecretStr("")
    rate_limit_max_per_ip: int = 60
    rate_limit_window_seconds: int = 60
    trust_forwarded_headers: bool = False


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância única (cacheada) das configurações.

    Returns:
        Settings: Configurações carregadas do ambiente.
    """
    return Settings()
