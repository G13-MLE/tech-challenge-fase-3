"""Pipeline de pré-processamento dos dados.

Carrega os laudos brutos de `data/raw`, aplica limpeza e normalização
de texto e salva os dados processados em `data/processed`.
"""

import csv
import logging
from pathlib import Path

from src.core.dataset import (
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    LABEL_COLUMN,
    TEXT_COLUMN,
    save_csv_records,
)
from src.preprocess.normalizer import (
    LowercaseNormalizer,
    PunctuationNormalizer,
    TextNormalizer,
    compose_normalizers,
)

logger = logging.getLogger(__name__)


def load_raw_records(path: Path) -> list[tuple[str, int]]:
    """Carrega os registros brutos do CSV de laudos.

    Args:
        path: Caminho do CSV bruto com as colunas text e label.

    Returns:
        Lista de tuplas (texto, rótulo de urgência).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados brutos não encontrado: {path}")
    records: list[tuple[str, int]] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append((row[TEXT_COLUMN], int(row[LABEL_COLUMN])))
    return records


def preprocess_text(text: str, normalizer: TextNormalizer) -> str:
    """Normaliza o texto de um laudo médico.

    Args:
        text: Texto bruto do laudo.
        normalizer: Estratégia de normalização a ser aplicada.

    Returns:
        Texto normalizado.
    """
    return normalizer.normalize(text)


def build_default_normalizer() -> TextNormalizer:
    """Constrói a estratégia padrão de normalização de texto.

    Returns:
        Estratégia composta (minúsculas sem acento + sem pontuação).
    """
    return compose_normalizers(LowercaseNormalizer(), PunctuationNormalizer())


def main() -> None:
    """Executa o pipeline de pré-processamento completo."""
    normalizer = build_default_normalizer()
    raw_records = load_raw_records(DATA_RAW_PATH)
    processed_records = [(preprocess_text(text, normalizer), label) for text, label in raw_records]
    save_csv_records(processed_records, DATA_PROCESSED_PATH)
    logger.info(
        "Pré-processamento concluído: %d laudos salvos em %s",
        len(processed_records),
        DATA_PROCESSED_PATH,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
