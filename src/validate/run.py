"""Validação dos dados brutos antes do pré-processamento.

Verifica se o CSV bruto tem as colunas esperadas (`text`, `label`),
se os rótulos pertencem a {0, 1, 2} e se não há textos vazios.
Falha com mensagem clara em caso de problema.
"""

import csv
import logging
from pathlib import Path

from src.core.dataset import DATA_RAW_PATH, EXPECTED_CLASSES, LABEL_COLUMN, TEXT_COLUMN

logger = logging.getLogger(__name__)

__all__ = ["validate_raw_data", "main"]


def validate_raw_data(path: Path = DATA_RAW_PATH) -> tuple[list[str], list[int]]:
    """Valida e carrega os dados brutos do CSV de laudos.

    Args:
        path: Caminho do CSV bruto com as colunas text e label.

    Returns:
        Tupla (textos, rótulos) validada.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se colunas ausentes, rótulos inválidos ou textos vazios.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados brutos não encontrado: {path}")
    texts: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or TEXT_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Coluna '{TEXT_COLUMN}' ausente no CSV. Colunas esperadas: "
                f"[{TEXT_COLUMN}, {LABEL_COLUMN}]"
            )
        if LABEL_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Coluna '{LABEL_COLUMN}' ausente no CSV. Colunas esperadas: "
                f"[{TEXT_COLUMN}, {LABEL_COLUMN}]"
            )
        for line_number, row in enumerate(reader, start=2):
            text = (row.get(TEXT_COLUMN) or "").strip()
            if not text:
                raise ValueError(f"Texto vazio encontrado na linha {line_number} do CSV.")
            try:
                label = int(row[LABEL_COLUMN])
            except TypeError, ValueError:
                raise ValueError(
                    f"Rótulo inválido '{row[LABEL_COLUMN]}' na linha {line_number}: "
                    f"esperado inteiro em {EXPECTED_CLASSES}."
                ) from None
            if label not in EXPECTED_CLASSES:
                raise ValueError(
                    f"Rótulo {label} na linha {line_number} fora do esperado {EXPECTED_CLASSES}."
                )
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError("CSV bruto não contém nenhum registro válido.")
    logger.info("Validação OK: %d registros com colunas e rótulos válidos.", len(texts))
    return texts, labels


def main() -> None:
    """Executa a validação dos dados brutos."""
    validate_raw_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
