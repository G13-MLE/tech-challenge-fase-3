"""Carregamento centralizado dos parâmetros do pipeline.

Lê o arquivo `configs/params.yaml` e disponibiliza os valores como
modelos Pydantic tipados para os estágios do pipeline.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

__all__ = ["PARAMS_PATH", "TrainParams", "Params", "load_params"]

PARAMS_PATH = Path("configs/params.yaml")


class TrainParams(BaseModel):
    """Parâmetros do estágio de treinamento.

    Attributes:
        seed: Seed fixa para reprodutibilidade.
        test_size: Proporção do conjunto de teste.
        classifier: Tipo de classificador ('random_forest' ou 'logistic_regression').
        random_forest_n_estimators: Número de árvores do RandomForest.
        random_forest_class_weight: Estratégia de peso das classes do RandomForest
            ('balanced' ou None).
        logistic_regression_class_weight: Estratégia de peso das classes da
            regressão logística ('balanced' ou None).
        logistic_regression_C: Inverso da regularização da regressão logística.
        logistic_regression_max_iter: Número máximo de iterações da regressão logística.
        tfidf_max_features: Número máximo de features do TF-IDF.
        tfidf_ngram_range: Faixa de n-grams (min, max) do TF-IDF.
        tfidf_sublinear_tf: Se True, aplica log1p na frequência de termos.
        tfidf_min_df: Frequência mínima de documentos para um termo.
            Inteiro = contagem absoluta; float = fração dos documentos.
        tfidf_max_df: Frequência máxima de documentos para um termo.
            Inteiro = contagem absoluta; float = fração dos documentos.
        tfidf_stopwords: Se True, usa a lista de stopwords em inglês.
            Alternativamente, pode ser uma lista de stopwords customizada.
        cv_folds: Número de folds para cross-validation (0 desativa).
        threshold_tuning: Se True, ajusta limiares de decisão por classe
            via cross-validation para maximizar recall macro.
        threshold_min_f1: F1 macro mínimo ao buscar limiares (trade-off recall vs precisão).
    """

    seed: int = Field(ge=0)
    test_size: float = Field(gt=0, lt=1)
    classifier: Literal["random_forest", "logistic_regression"] = "random_forest"
    random_forest_n_estimators: int = Field(ge=1)
    random_forest_class_weight: str | None = "balanced"
    logistic_regression_class_weight: str | None = "balanced"
    logistic_regression_C: float = Field(gt=0, default=1.0)  # noqa: N815
    logistic_regression_max_iter: int = Field(ge=100, default=1000)
    tfidf_max_features: int = Field(ge=1)
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_sublinear_tf: bool = True
    tfidf_min_df: int | float = 2
    tfidf_max_df: int | float = 0.95
    tfidf_stopwords: bool | list[str] = False
    cv_folds: int = Field(ge=0, default=5)
    threshold_tuning: bool = True
    threshold_min_f1: float = Field(ge=0.0, le=1.0, default=0.55)


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
