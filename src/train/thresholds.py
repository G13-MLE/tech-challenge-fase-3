"""Ajuste de limiares de decisão por classe para maximizar recall.

Utiliza predições out-of-fold de cross-validation para buscar limiares
de decisão (offsets por classe) que maximizem o recall macro, mantendo
F1 macro acima de um mínimo especificado.

Os limiares são salvos como artefato JSON e aplicados no momento de
inferência, ajustando as probabilidades preditas antes do argmax.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import cross_val_predict

from src.core.dataset import THRESHOLDS_PATH, atomic_write_text
from src.core.params import TrainParams

__all__ = [
    "DEFAULT_THRESHOLDS",
    "tune_thresholds",
    "save_thresholds",
    "load_thresholds",
    "apply_thresholds",
]

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {"0": 0.0, "1": 0.0, "2": 0.0}


def tune_thresholds(
    pipeline,
    texts: list[str],
    labels: list[int],
    params: TrainParams,
) -> dict[str, float]:
    """Busca limiares de decisão por classe via cross-validation.

    Usa cross_val_predict para obter probabilidades out-of-fold e
    busca em grade os offsets por classe que maximizam o recall macro,
    mantendo F1 macro >= params.threshold_min_f1.

    Args:
        pipeline: Pipeline sklearn não treinado.
        texts: Textos de treino.
        labels: Rótulos de treino.
        params: Parâmetros do pipeline (usa cv_folds e threshold_min_f1).

    Returns:
        Dicionário com offsets por classe (chaves "0", "1", "2").
    """
    cv_folds = params.cv_folds if params.cv_folds > 0 else 3
    min_f1 = params.threshold_min_f1

    class_counts = {label: labels.count(label) for label in set(labels)}
    min_class_count = min(class_counts.values())
    if cv_folds > min_class_count:
        cv_folds = min_class_count

    probas = cross_val_predict(pipeline, texts, labels, cv=cv_folds, method="predict_proba")
    probas = np.asarray(probas)

    p0 = probas[:, 0]
    p1 = probas[:, 1]
    p2 = probas[:, 2]

    best_recall = -1.0
    best_offsets = (0.0, 0.0, 0.0)

    for t0 in np.arange(-0.5, 1.5, 0.025):
        for t2 in np.arange(-0.5, 1.5, 0.025):
            scores = np.column_stack([p0 + t0, p1, p2 + t2])
            preds = np.argmax(scores, axis=1)
            macro_recall = recall_score(labels, preds, average="macro")
            macro_f1 = f1_score(labels, preds, average="macro")
            if macro_recall > best_recall and macro_f1 >= min_f1:
                best_recall = macro_recall
                best_offsets = (float(t0), 0.0, float(t2))

    logger.info(
        "Limiares ajustados: t0=%.3f, t1=%.3f, t2=%.3f (CV recall macro=%.4f)",
        best_offsets[0],
        best_offsets[1],
        best_offsets[2],
        best_recall,
    )

    return {
        "0": round(best_offsets[0], 4),
        "1": round(best_offsets[1], 4),
        "2": round(best_offsets[2], 4),
    }


def save_thresholds(thresholds: dict[str, float], path: Path = THRESHOLDS_PATH) -> None:
    """Persiste os limiares em JSON de forma atômica.

    Args:
        thresholds: Dicionário com offsets por classe.
        path: Caminho do arquivo JSON de saída.
    """
    content = json.dumps(thresholds, indent=2) + "\n"
    atomic_write_text(path, content)


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, float]:
    """Carrega os limiares de um arquivo JSON.

    Se o arquivo não existir, retorna limiares neutros (0.0 para todas as classes),
    permitindo que o sistema funcione sem ajuste de limiares.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Dicionário com offsets por classe (chaves "0", "1", "2").
    """
    if not path.exists():
        logger.debug("Arquivo de limiares não encontrado em %s — usando neutros.", path)
        return dict(DEFAULT_THRESHOLDS)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return {str(k): float(v) for k, v in data.items()}


def apply_thresholds(
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    classes: list[int],
) -> np.ndarray:
    """Aplica offsets de limiares nas probabilidades e retorna índices argmax.

    Args:
        probabilities: Array de probabilidades (n_samples, n_classes).
        thresholds: Dicionário com offsets por classe.
        classes: Lista de classes do modelo (ex.: [0, 1, 2]).

    Returns:
        Array de rótulos preditos (n_samples,).
    """
    adjusted = probabilities.copy()
    for idx, cls in enumerate(classes):
        key = str(int(cls))
        if key in thresholds:
            adjusted[:, idx] += thresholds[key]
    return np.argmax(adjusted, axis=1)
