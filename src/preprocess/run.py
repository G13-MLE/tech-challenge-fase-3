"""Pipeline de pré-processamento dos dados.

Carrega os laudos brutos de `data/raw`, valida os dados, aplica
limpeza e normalização de texto e salva os dados processados em
`data/processed`.
"""

import logging
from pathlib import Path

from src.core.dataset import (
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    save_csv_records,
)
from src.preprocess.normalizer import (
    TextNormalizer,
    build_default_normalizer,
)
from src.validate.run import validate_raw_data

logger = logging.getLogger(__name__)


def load_raw_records(path: Path) -> list[tuple[str, int]]:
    """Carrega os registros brutos do CSV de laudos.

    Args:
        path: Caminho do CSV bruto com as colunas text e label.

    Returns:
        Lista de tuplas (texto, rótulo de urgência).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se os dados brutos falharem na validação.
    """
    texts, labels = validate_raw_data(path)
    return list(zip(texts, labels, strict=True))


def preprocess_text(text: str, normalizer: TextNormalizer) -> str:
    """Normaliza o texto de um laudo médico.

    Args:
        text: Texto bruto do laudo.
        normalizer: Estratégia de normalização a ser aplicada.

    Returns:
        Texto normalizado.
    """
    return normalizer.normalize(text)


def main() -> None:
    """Executa o pipeline de pré-processamento completo.

    Primeiro valida os dados brutos (colunas, rótulos, textos não vazios)
    e depois aplica a normalização de texto.
    """
    normalizer = build_default_normalizer()
    texts, labels = validate_raw_data(DATA_RAW_PATH)
    raw_records = list(zip(texts, labels, strict=True))
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
