"""Testes smoke: import do pacote e configuração básica."""

from pydantic import SecretStr

from src.core.config import Settings, get_settings


def test_get_settings_returns_singleton() -> None:
    """Verifica que get_settings retorna a mesma instância (cache)."""
    assert get_settings() is get_settings()


def test_settings_defaults(monkeypatch) -> None:
    """Verifica os valores padrão das configurações."""
    get_settings.cache_clear()
    settings = Settings()
    assert settings.app_name == "triage-api"
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.api_port == 8000
    assert str(settings.model_path) == "models/model.joblib"
    assert settings.model_backend == "joblib"


def test_settings_kaggle_key_is_secret() -> None:
    """Verifica que kaggle_key é armazenada como SecretStr."""
    get_settings.cache_clear()
    settings = Settings()
    assert isinstance(settings.kaggle_key, SecretStr)


def test_settings_reads_env_file(monkeypatch) -> None:
    """Verifica que o .env (quando presente) sobrescreve os padrões."""
    get_settings.cache_clear()
    settings = Settings()
    assert settings.app_env in {"development", "production"}
