"""Validação do ambiente de desenvolvimento.

Verifica a existência do arquivo `.env`, a versão do Python, a
importabilidade das dependências essenciais e a presença dos diretórios
obrigatórios do projeto. Imprime um resumo [OK]/[ERROR] e sai com
código 1 se qualquer validação falhar.
"""

import importlib
import sys
from pathlib import Path

REQUIRED_DIRS = [
    "data/raw",
    "models",
    "configs",
    "scripts",
    "docker",
    "tests",
]

REQUIRED_PACKAGES = [
    "fastapi",
    "sklearn",
    "prometheus_client",
]


def check_env_file() -> bool:
    """Verifica se o arquivo `.env` existe.

    Returns:
        bool: True se o arquivo existe, False caso contrário.
    """
    if Path(".env").exists():
        print("[OK] .env encontrado")
        return True
    print("[ERROR] .env não encontrado. Copie de .env.example: cp .env.example .env")
    return False


def check_python_version() -> bool:
    """Verifica se a versão do Python é 3.14.x.

    Returns:
        bool: True se a versão é 3.14.x, False caso contrário.
    """
    version = sys.version_info
    if version.major == 3 and version.minor == 14:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    print(f"[ERROR] Python {version.major}.{version.minor}.{version.micro} (esperado 3.14.x)")
    return False


def check_packages() -> bool:
    """Verifica se as dependências essenciais são importáveis.

    Returns:
        bool: True se todas as dependências importam, False caso contrário.
    """
    ok = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"[OK] dependência {package} importável")
        except ImportError:
            print(f"[ERROR] dependência {package} não importável")
            ok = False
    return ok


def check_directories() -> bool:
    """Verifica se os diretórios obrigatórios existem.

    Returns:
        bool: True se todos os diretórios existem, False caso contrário.
    """
    ok = True
    for directory in REQUIRED_DIRS:
        if Path(directory).is_dir():
            print(f"[OK] diretório {directory}/ existe")
        else:
            print(f"[ERROR] diretório {directory}/ não existe")
            ok = False
    return ok


def check_dvc_remote() -> bool:
    """Verifica se o remote DVC está configurado (opcional).

    O remote OneDrive é opcional: o pipeline roda self-contained sem ele.
    Apenas emite um aviso, sem falhar a validação.

    Returns:
        bool: Sempre True (aviso não bloqueia).
    """
    env_path = Path(".env")
    if not env_path.exists():
        return True
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DVC_ONEDRIVE_REMOTE_URL=") and line.split("=", 1)[1].strip():
            print("[OK] remote DVC OneDrive configurado")
            return True
    print("[WARN] DVC_ONEDRIVE_REMOTE_URL vazio — remote opcional, pipeline roda self-contained")
    return True


def main() -> None:
    """Executa todas as validações e define o código de saída."""
    checks = [
        check_env_file(),
        check_python_version(),
        check_packages(),
        check_directories(),
        check_dvc_remote(),
    ]
    if all(checks):
        print("\n[OK] Ambiente validado com sucesso")
        sys.exit(0)
    print("\n[ERROR] Falhas na validação do ambiente")
    sys.exit(1)


if __name__ == "__main__":
    main()
