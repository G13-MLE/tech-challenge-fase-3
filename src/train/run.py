"""Pipeline de treinamento do modelo de triagem.

Carrega os dados processados, treina um Pipeline TF-IDF + classificador
(RandomForest ou LogisticRegression) com seeds fixos e salva o artefato
em `models/model.joblib`.
"""

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.core.dataset import (
    DATA_PROCESSED_PATH,
    EXPECTED_CLASSES,
    FEATURE_IMPORTANCES_PATH,
    MODEL_HASH_PATH,
    MODEL_PATH,
    TEST_DATA_PATH,
    TRAIN_METRICS_PATH,
    atomic_write_bytes,
    atomic_write_text,
    load_csv_records,
    save_csv_records,
)
from src.core.params import TrainParams, load_params
from src.core.stopwords import ENGLISH_STOPWORDS

logger = logging.getLogger(__name__)


def resolve_stopwords(params: TrainParams) -> list[str] | None:
    """Resolve a configuração de stopwords do TF-IDF.

    Args:
        params: Parâmetros do pipeline.

    Returns:
        Lista de stopwords ou None para desativar.
    """
    if isinstance(params.tfidf_stopwords, list):
        return params.tfidf_stopwords
    if params.tfidf_stopwords is True:
        return ENGLISH_STOPWORDS
    return None


def build_pipeline(params: TrainParams) -> Pipeline:
    """Constrói o Pipeline TF-IDF + classificador.

    O classificador é selecionado pelo parâmetro `classifier`:
    - 'random_forest': RandomForestClassifier (padrão)
    - 'logistic_regression': LogisticRegression

    Args:
        params: Parâmetros do pipeline (seeds, TF-IDF, classificador).

    Returns:
        Pipeline sklearn não treinado.
    """
    stopwords = resolve_stopwords(params)
    tfidf_kwargs: dict[str, Any] = {
        "max_features": params.tfidf_max_features,
        "ngram_range": params.tfidf_ngram_range,
        "sublinear_tf": params.tfidf_sublinear_tf,
        "min_df": params.tfidf_min_df,
        "max_df": params.tfidf_max_df,
    }
    if stopwords is not None:
        tfidf_kwargs["stop_words"] = stopwords

    if params.classifier == "logistic_regression":
        clf = LogisticRegression(
            C=params.logistic_regression_C,
            max_iter=params.logistic_regression_max_iter,
            random_state=params.seed,
            class_weight=params.logistic_regression_class_weight,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=params.random_forest_n_estimators,
            random_state=params.seed,
            class_weight=params.random_forest_class_weight,
        )

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", clf),
        ]
    )


def save_model(model: Pipeline, path: Path) -> None:
    """Persiste o modelo treinado de forma atômica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    atomic_write_bytes(path, buf.getvalue())


def save_model_hash(model_path: Path, hash_path: Path) -> None:
    """Calcula e salva o hash SHA256 do modelo para verificação de integridade.

    Args:
        model_path: Caminho do arquivo de modelo.
        hash_path: Caminho do arquivo de hash de saída.
    """
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    atomic_write_text(hash_path, sha256)


def validate_model_classes(model: Pipeline) -> None:
    """Valida que as classes do modelo correspondem ao mapeamento esperado.

    Args:
        model: Pipeline sklearn treinado.

    Raises:
        ValueError: Se as classes não correspondem ao esperado.
    """
    actual = list(model.classes_)
    if actual != EXPECTED_CLASSES:
        raise ValueError(
            f"Classes do modelo {actual} não correspondem ao esperado {EXPECTED_CLASSES}"
        )


def save_feature_importances(model: Pipeline, path: Path) -> None:
    """Salva as top-N importâncias de features do classificador.

    Para RandomForest, usa feature_importances_.
    Para LogisticRegression, usa os coeficientes absolutos por classe.

    Args:
        model: Pipeline sklearn treinado.
        path: Caminho do CSV de saída.
    """
    clf = model.named_steps["clf"]
    tfidf = model.named_steps["tfidf"]
    feature_names = tfidf.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = abs(clf.coef_).mean(axis=0)
    else:
        logger.warning("Classificador não suporta importâncias de features.")
        return

    top_features = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    content = "feature,importance\n" + "\n".join(f"{name},{imp:.8f}" for name, imp in top_features)
    atomic_write_text(path, content + "\n")


def save_train_metrics(metrics: dict[str, float], path: Path) -> None:
    """Persiste métricas do treino (ex.: CV) em JSON de forma atômica.

    Args:
        metrics: Dicionário de métricas.
        path: Caminho do arquivo JSON de saída.
    """
    content = json.dumps(metrics, indent=2) + "\n"
    atomic_write_text(path, content)


def run_cross_validation(
    pipeline: Pipeline, texts: list[str], labels: list[int], cv_folds: int
) -> dict[str, float]:
    """Executa cross-validation opcional e retorna as métricas.

    Se o número de folds solicitado for maior que o número de membros
    de alguma classe (limite do stratified CV), reduz os folds para o
    máximo viável com warning — o pipeline nunca falha por causa disso.

    Args:
        pipeline: Pipeline sklearn.
        texts: Textos de treino.
        labels: Rótulos de treino.
        cv_folds: Número de folds (0 desativa).

    Returns:
        Dicionário com `cv_f1_macro_mean` e `cv_f1_macro_std`, ou vazio se desativado.
    """
    if cv_folds <= 0:
        return {}
    class_counts = {label: labels.count(label) for label in set(labels)}
    min_class_count = min(class_counts.values())
    if cv_folds > min_class_count:
        logger.warning(
            "cv_folds=%d maior que a menor classe (%d) — reduzindo CV para %d folds.",
            cv_folds,
            min_class_count,
            min_class_count,
        )
        cv_folds = min_class_count
    scores = cross_val_score(pipeline, texts, labels, cv=cv_folds, scoring="f1_macro")
    return {
        "cv_f1_macro_mean": float(scores.mean()),
        "cv_f1_macro_std": float(scores.std()),
    }


def train_and_save(
    data_path: Path = DATA_PROCESSED_PATH,
    model_path: Path = MODEL_PATH,
    model_hash_path: Path = MODEL_HASH_PATH,
    test_data_path: Path = TEST_DATA_PATH,
    feature_importances_path: Path | None = None,
    train_metrics_path: Path | None = None,
    params: TrainParams | None = None,
) -> Path:
    """Executa o pipeline de treinamento completo e salva os artefatos.

    Carrega dados processados, divide em treino/teste com seed fixo,
    treina o Pipeline TF-IDF + classificador, valida as classes, salva
    modelo + hash + split de teste + importâncias + métricas de CV.

    Args:
        data_path: Caminho do CSV processado.
        model_path: Caminho do arquivo de modelo.
        model_hash_path: Caminho do arquivo de hash SHA256.
        test_data_path: Caminho do CSV de teste.
        feature_importances_path: Caminho do CSV de importâncias. Se None, usa o padrão.
        train_metrics_path: Caminho do JSON de métricas de treino. Se None, usa o padrão.
        params: Parâmetros do pipeline. Se None, carrega de configs/params.yaml.

    Returns:
        Caminho do modelo salvo.
    """
    if params is None:
        params = load_params().train
    if feature_importances_path is None:
        feature_importances_path = FEATURE_IMPORTANCES_PATH
    if train_metrics_path is None:
        train_metrics_path = TRAIN_METRICS_PATH

    texts, labels = load_csv_records(data_path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=params.test_size,
        random_state=params.seed,
        stratify=labels,
    )
    pipeline = build_pipeline(params)
    pipeline.fit(train_texts, train_labels)
    validate_model_classes(pipeline)
    save_model(pipeline, model_path)
    save_model_hash(model_path, model_hash_path)
    save_csv_records(list(zip(test_texts, test_labels, strict=True)), test_data_path)
    save_feature_importances(pipeline, feature_importances_path)
    cv_metrics = run_cross_validation(pipeline, train_texts, train_labels, params.cv_folds)
    if cv_metrics:
        save_train_metrics(cv_metrics, train_metrics_path)
    logger.info("Treinamento concluído: modelo salvo em %s", model_path)
    logger.info("Split de teste salvo em %s", test_data_path)
    return model_path


def main() -> None:
    """Executa o pipeline de treinamento completo."""
    train_and_save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
