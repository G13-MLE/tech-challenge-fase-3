"""Cria o modelo dummy para a API de triagem.

Gera um Pipeline sklearn (TfidfVectorizer + RandomForestClassifier)
treinado com textos sintéticos que representam as 3 classes de urgência:
- normal: asma, hérnia
- atenção: diabetes, hipertensão
- urgente: pneumonia

O modelo é salvo em models/model.joblib para ser servido pela API.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.models.urgency import URGENCY_MAP, UrgencyLevel

CLASS_NORMAL = 0
CLASS_ATENCAO = 1
CLASS_URGENTE = 2

TRAINING_DATA = [
    ("paciente com asma brônquica apresenta sibilos e dispneia", CLASS_NORMAL),
    ("crise asmática com broncoespasmo e tosse seca", CLASS_NORMAL),
    ("asma crônica em tratamento com inalador", CLASS_NORMAL),
    ("exacerbação de asma com dificuldade respiratória leve", CLASS_NORMAL),
    ("hérnia inguinal unilateral sem estrangulamento", CLASS_NORMAL),
    ("hérnia umbilical redutível sem sinais de complicação", CLASS_NORMAL),
    ("hérnia hiatal com refluxo gastroesofágico", CLASS_NORMAL),
    ("correção de hérnia ventral programada", CLASS_NORMAL),
    ("paciente diabético com glicemia elevada e poliúria", CLASS_ATENCAO),
    ("diabetes tipo 2 descompensada com hemoglobina glicada alta", CLASS_ATENCAO),
    ("diabetes mellitus com neuropatia periférica", CLASS_ATENCAO),
    ("hipertensão arterial crônica com pressão acima de 140x90", CLASS_ATENCAO),
    ("crise hipertensiva com cefaleia e tontura", CLASS_ATENCAO),
    ("hipertensão essencial em tratamento com losartana", CLASS_ATENCAO),
    ("paciente com pneumonia comunitária com infiltrado bilateral", CLASS_URGENTE),
    ("pneumonia bacteriana grave com insuficiência respiratória", CLASS_URGENTE),
    ("pneumonia nosocomial em paciente internado", CLASS_URGENTE),
    ("pneumonia lobar com febre alta e dispneia intensa", CLASS_URGENTE),
    ("abscesso pulmonar com pneumonia associada", CLASS_URGENTE),
]

VALIDATION_CASES = [
    ("paciente com pneumonia grave", UrgencyLevel.URGENTE),
    ("diabetes tipo 2 descompensada", UrgencyLevel.ATENCAO),
    ("crise asmática com broncoespasmo", UrgencyLevel.NORMAL),
    ("hérnia inguinal redutível", UrgencyLevel.NORMAL),
    ("hipertensão arterial crônica", UrgencyLevel.ATENCAO),
]

SEED = 42
MODEL_OUTPUT_PATH = Path("models/model.joblib")


def create_dummy_model() -> Pipeline:
    """Cria e treina o Pipeline sklearn com dados sintéticos.

    Returns:
        Pipeline sklearn treinado.
    """
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]
    pipeline = _build_pipeline()
    pipeline.fit(texts, labels)
    return pipeline


def _build_pipeline() -> Pipeline:
    """Constrói o Pipeline sklearn com TF-IDF e RandomForest.

    Returns:
        Pipeline sklearn não treinado.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ]
    )


def validate_model(model: Pipeline) -> None:
    """Valida o modelo contra textos conhecidos.

    Args:
        model: Pipeline sklearn treinado.

    Raises:
        RuntimeError: Se alguma predição estiver incorreta.
    """
    errors = _collect_validation_errors(model)
    if errors:
        error_msg = "\n".join(errors)
        raise RuntimeError(f"Validação do modelo falhou:\n{error_msg}")


def _collect_validation_errors(model: Pipeline) -> list[str]:
    """Coleta erros de validação do modelo.

    Args:
        model: Pipeline sklearn treinado.

    Returns:
        Lista de mensagens de erro, vazia se todas as predições estão corretas.
    """
    errors = []
    for text, expected in VALIDATION_CASES:
        predicted_class = model.predict([text])[0]
        predicted_urgency = URGENCY_MAP[predicted_class]
        if predicted_urgency != expected:
            errors.append(
                f"Esperado '{expected.value}', obtido '{predicted_urgency.value}' para: {text}"
            )
    return errors


def main() -> None:
    """Cria, valida e salva o modelo dummy."""
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Criando modelo dummy...")
    model = create_dummy_model()

    print("Validando predições...")
    validate_model(model)

    print(f"Salvando modelo em {MODEL_OUTPUT_PATH}...")
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print("[OK] Modelo dummy criado com sucesso.")


if __name__ == "__main__":
    main()
