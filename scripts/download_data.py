"""Download do Medical Abstracts TC Corpus do Kaggle.

Baixa o dataset `saharalaa/medical-abstracts-tc-corpus` via kagglehub,
mapeia as 5 classes clínicas para 3 níveis de urgência e salva o CSV
em `data/raw/medical_abstracts.csv`.

Mapeamento classe clínica → urgência (Path B):
  - pneumonia → urgente  (2)
  - diabetes, hipertensão → atenção (1)
  - asma, hérnia → normal (0)

Uso:
    uv run python scripts/download_data.py
    KAGGLE_USERNAME=user KAGGLE_KEY=key uv run python scripts/download_data.py
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "saharalaa/medical-abstracts-tc-corpus"
RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = RAW_DATA_DIR / "medical_abstracts.csv"

CLINICAL_TO_URGENCY: dict[str, int] = {
    "Asthma": 0,
    "Diabetes": 1,
    "Hernia": 0,
    "Hypertension": 1,
    "Pneumonia": 2,
}

URGENCY_NAMES: dict[int, str] = {
    0: "normal",
    1: "atenção",
    2: "urgente",
}

CONDITION_FILES: dict[str, str] = {
    "Asthma": "Asthma.csv",
    "Diabetes": "Diabetes.csv",
    "Hernia": "Hernia.csv",
    "Hypertension": "Hypertension.csv",
    "Pneumonia": "Pneumonia.csv",
}


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

        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "--unzip"],
            capture_output=True,
            text=True,
            check=True,
        )
        import tempfile

        path = tempfile.mkdtemp()
        logger.info("Dataset baixado via kaggle CLI: %s", result.stdout)
        return path
    except Exception as exc2:
        raise RuntimeError(
            "Falha ao baixar o dataset. Configure KAGGLE_USERNAME/KAGGLE_KEY "
            f"ou ~/.kaggle/kaggle.json. Erros: kagglehub={kagglehub_error}, kaggle={exc2}"
        ) from None


def load_condition_csv(path: Path, condition: str) -> list[tuple[str, int]]:
    """Lê um CSV de condição médica e retorna (texto, label_urgência).

    Procura a coluna 'abstract' no CSV. Se não encontrar, usa a segunda coluna.

    Args:
        path: Caminho do CSV da condição.
        condition: Nome da condição (chave em CLINICAL_TO_URGENCY).

    Returns:
        Lista de tuplas (texto, label_urgência).
    """
    urgency = CLINICAL_TO_URGENCY[condition]
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row.get("abstract", "")
            if not text:
                values = list(row.values())
                if len(values) >= 2:
                    text = values[1]
            if text.strip():
                rows.append((text.strip(), urgency))
    return rows


def merge_conditions(dataset_dir: str) -> list[tuple[str, int]]:
    """Lê todos os CSVs de condições e retorna registros mesclados.

    Args:
        dataset_dir: Diretório onde o dataset foi baixado.

    Returns:
        Lista de tuplas (texto, label_urgência).
    """
    dataset_path = Path(dataset_dir)
    all_rows: list[tuple[str, int]] = []

    for condition, filename in CONDITION_FILES.items():
        csv_path = dataset_path / filename
        if not csv_path.exists():
            candidates = list(dataset_path.glob(f"**/{filename}"))
            if candidates:
                csv_path = candidates[0]
            else:
                candidates = list(dataset_path.glob(f"**/{condition}.csv"))
                if candidates:
                    csv_path = candidates[0]
                else:
                    logger.warning("Arquivo não encontrado para %s: %s", condition, filename)
                    continue

        condition_rows = load_condition_csv(csv_path, condition)
        urgency = CLINICAL_TO_URGENCY[condition]
        logger.info(
            "  %s → %d registros (urgência=%s)",
            condition,
            len(condition_rows),
            URGENCY_NAMES[urgency],
        )
        all_rows.extend(condition_rows)

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

    from collections import Counter

    counts = Counter(label for _, label in rows)
    logger.info("Dataset salvo em %s: %d registros", output_path, len(rows))
    for label in sorted(counts):
        logger.info("  Classe %d (%s): %d registros", label, URGENCY_NAMES[label], counts[label])


def main() -> None:
    """Baixa o dataset do Kaggle e salva em data/raw/medical_abstracts.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Baixando dataset %s ...", KAGGLE_DATASET)
    dataset_dir = download_kaggle_dataset()

    logger.info("Mesclando condições clínicas em níveis de urgência...")
    rows = merge_conditions(dataset_dir)

    if not rows:
        raise RuntimeError("Nenhum registro encontrado no dataset baixado.")

    save_dataset(rows, OUTPUT_PATH)
    print(f"[OK] Dataset baixado e salvo: {len(rows)} registros em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
