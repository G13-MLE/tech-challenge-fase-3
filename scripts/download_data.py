"""Download do Medical Abstracts TC Corpus do Kaggle.

Baixa o dataset `saharalaa/medical-abstracts-tc-corpus` via kagglehub,
mapeia as 5 classes clínicas para 3 níveis de urgência e salva o CSV
em `data/raw/laudos.csv`.

Mapeamento condition_label → urgência:
  - 1 (neoplasms) → urgente  (2)
  - 3 (nervous system diseases) → atenção (1)
  - 4 (cardiovascular diseases) → atenção (1)
  - 2 (digestive system diseases) → normal (0)
  - 5 (general pathological conditions) → atenção (1)

Uso:
    uv run python scripts/download_data.py
    KAGGLE_USERNAME=user KAGGLE_KEY=key uv run python scripts/download_data.py
"""

import csv
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CLINICAL_TO_URGENCY",
    "URGENCY_NAMES",
    "LABEL_NAMES",
    "CONDITION_LABEL_TO_URGENCY",
    "download_kaggle_dataset",
    "load_medical_csvs",
    "save_dataset",
    "main",
]

KAGGLE_DATASET = "saharalaa/medical-abstracts-tc-corpus"
RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = RAW_DATA_DIR / "laudos.csv"

CONDITION_LABEL_TO_URGENCY: dict[int, int] = {
    1: 2,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
}

LABEL_NAMES: dict[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

CLINICAL_TO_URGENCY: dict[str, int] = {
    "neoplasms": 2,
    "nervous system diseases": 1,
    "cardiovascular diseases": 1,
    "digestive system diseases": 0,
    "general pathological conditions": 1,
}

URGENCY_NAMES: dict[int, str] = {
    0: "normal",
    1: "atenção",
    2: "urgente",
}

TRAIN_CSV = "medical_tc_train.csv"
TEST_CSV = "medical_tc_test.csv"


def download_kaggle_dataset() -> str:
    """Baixa o dataset do Kaggle via kagglehub.

    Tenta primeiro com kagglehub (credenciais via env ou ~/.kaggle/kaggle.json).
    Se falhar, tenta via kaggle CLI como fallback.

    Returns:
        Caminho local do diretório com os arquivos CSV.
    """
    kagglehub_error: str = ""
    try:
        import kagglehub

        path = kagglehub.dataset_download(KAGGLE_DATASET)
        logger.info("Dataset baixado via kagglehub em: %s", path)
        return path
    except Exception as exc:
        kagglehub_error = str(exc)
        logger.warning("kagglehub falhou (%s), tentando kaggle CLI...", exc)

    try:
        import subprocess
        import tempfile

        path = tempfile.mkdtemp()
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", path, "--unzip"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Dataset baixado via kaggle CLI: %s", result.stdout)
        return path
    except Exception as exc2:
        raise RuntimeError(
            "Falha ao baixar o dataset. Configure KAGGLE_USERNAME/KAGGLE_KEY "
            f"ou ~/.kaggle/kaggle.json. Erros: kagglehub={kagglehub_error}, kaggle={exc2}"
        ) from None


def _find_csv(dataset_path: Path, filename: str) -> Path:
    """Localiza um CSV no diretório do dataset, incluindo subdiretórios.

    Args:
        dataset_path: Diretório raiz do dataset baixado.
        filename: Nome do arquivo CSV a localizar.

    Returns:
        Caminho completo do arquivo encontrado.

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
    """
    direct = dataset_path / filename
    if direct.exists():
        return direct
    candidates = list(dataset_path.glob(f"**/{filename}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Arquivo {filename} não encontrado em {dataset_path}")


def load_medical_csvs(dataset_dir: str) -> list[tuple[str, int]]:
    """Lê os CSVs de treino e teste e mapeia condition_label para urgência.

    Args:
        dataset_dir: Diretório onde o dataset foi baixado.

    Returns:
        Lista de tuplas (texto, label_urgência).
    """
    dataset_path = Path(dataset_dir)
    all_rows: list[tuple[str, int]] = []

    for csv_name in (TRAIN_CSV, TEST_CSV):
        csv_path = _find_csv(dataset_path, csv_name)
        logger.info("Lendo %s ...", csv_path.name)
        file_rows: int = 0
        with csv_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                condition_label = int(row["condition_label"].strip())
                text = row["medical_abstract"].strip()
                if not text:
                    continue
                urgency = CONDITION_LABEL_TO_URGENCY[condition_label]
                all_rows.append((text, urgency))
                file_rows += 1
        logger.info("  %s: %d registros", csv_path.name, file_rows)

    return all_rows


def save_dataset(rows: list[tuple[str, int]], output_path: Path) -> None:
    """Salva o dataset mesclado em CSV.

    Args:
        rows: Lista de tuplas (texto, label_urgência).
        output_path: Caminho do CSV de saída.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    counts = Counter(label for _, label in rows)
    logger.info("Dataset salvo em %s: %d registros", output_path, len(rows))
    for label in sorted(counts):
        logger.info("  Classe %d (%s): %d registros", label, URGENCY_NAMES[label], counts[label])


def main() -> None:
    """Baixa o dataset do Kaggle e salva em data/raw/laudos.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Baixando dataset %s ...", KAGGLE_DATASET)
    dataset_dir = download_kaggle_dataset()

    logger.info("Mapeando classes clínicas para níveis de urgência...")
    rows = load_medical_csvs(dataset_dir)

    if not rows:
        raise RuntimeError("Nenhum registro encontrado no dataset baixado.")

    save_dataset(rows, OUTPUT_PATH)
    print(f"[OK] Dataset baixado e salvo: {len(rows)} registros em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
