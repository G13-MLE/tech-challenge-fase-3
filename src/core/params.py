"""Carregamento centralizado dos parâmetros do pipeline.

Lê o arquivo `configs/params.yaml` e disponibiliza os valores como
modelos Pydantic tipados para os estágios do pipeline.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PARAMS_PATH = Path("configs/params.yaml")


class TrainParams(BaseModel):
    """Parâmetros do estágio de treinamento.

    Attributes:
        seed: Seed fixa para reprodutibilidade.
        test_size: Proporção do conjunto de teste.
        random_forest_n_estimators: Número de árvores do RandomForest.
        tfidf_max_features: Número máximo de features do TF-IDF.
    """

    seed: int = Field(ge=0)
    test_size: float = Field(gt=0, lt=1)
    random_forest_n_estimators: int = Field(ge=1)
    tfidf_max_features: int = Field(ge=1)


class Params(BaseModel):
    """Parâmetros do pipeline.

    Attributes:
        train: Parâmetros do estágio de treinamento.
    """

    train: TrainParams


@lru_cache
def load_params() -> Params:
    """Carrega os parâmetros do arquivo configs/params.yaml.

    O resultado é cacheado por invocação (lru_cache). Cada estágio
    do pipeline executa em processo separado, portanto o cache não
    causa problemas de staleness em execução normal. Não chame
    cache_clear() em código de produção.

    Returns:
        Params: Parâmetros tipados e validados do pipeline.

    Raises:
        FileNotFoundError: Se o arquivo de parâmetros não existir.
    """
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {PARAMS_PATH}")
    with PARAMS_PATH.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return Params(train=TrainParams(**raw["train"]))
