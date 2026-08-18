"""Gera o dataset sintético de laudos médicos para o pipeline.

Cria `data/raw/laudos.csv` com textos de laudos médicos sintéticos
representando as 3 classes de urgência:
- 0 (normal): asma, hérnia
- 1 (atenção): diabetes, hipertensão
- 2 (urgente): pneumonia

Os textos são combinações determinísticas de sujeito, condição e
contexto, com seed fixo para reprodutibilidade completa do dataset.
"""

import random
from pathlib import Path

from src.core.dataset import save_csv_records
from src.models.urgency import URGENCY_MAP

SEED = 42
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_PATH = RAW_DATA_DIR / "laudos.csv"

SUBJECTS = [
    "paciente",
    "paciente do sexo masculino",
    "paciente do sexo feminino",
    "paciente idoso",
    "paciente adulto jovem",
    "paciente em acompanhamento ambulatorial",
    "paciente internado",
]

CONDITIONS: dict[int, list[str]] = {
    0: [
        "asma brônquica apresenta sibilos e dispneia",
        "crise asmática com broncoespasmo e tosse seca",
        "asma crônica em tratamento com inalador",
        "exacerbação de asma com dificuldade respiratória leve",
        "hérnia inguinal unilateral sem estrangulamento",
        "hérnia umbilical redutível sem sinais de complicação",
        "hérnia hiatal com refluxo gastroesofágico",
        "hérnia ventral redutível com dor leve",
    ],
    1: [
        "diabetes tipo 2 com glicemia elevada e poliúria",
        "diabetes mellitus descompensada com hemoglobina glicada alta",
        "diabetes com neuropatia periférica",
        "hipertensão arterial crônica com pressão acima de 140x90",
        "hipertensão essencial em tratamento com losartana",
        "hipertensão com cefaleia e tontura",
    ],
    2: [
        "pneumonia comunitária com infiltrado bilateral",
        "pneumonia bacteriana grave com insuficiência respiratória",
        "pneumonia nosocomial em paciente internado",
        "pneumonia lobar com febre alta e dispneia intensa",
        "abscesso pulmonar com pneumonia associada",
    ],
}

CONTEXTS: dict[int, list[str]] = {
    0: [
        "evolução favorável",
        "sem sinais de complicação",
        "manejo ambulatorial",
        "sintomas controlados",
        "sem alterações relevantes",
    ],
    1: [
        "necessita ajuste de medicação",
        "acompanhamento clínico próximo",
        "controle metabólico insatisfatório",
        "revisão de conduta em consulta",
        "sinais de descompensação moderada",
    ],
    2: [
        "urgência médica",
        "internação imediata",
        "risco de insuficiência respiratória",
        "monitorização intensiva",
        "tratamento hospitalar emergencial",
    ],
}

ROWS_PER_CLASS = 60


def build_dataset() -> list[tuple[str, int]]:
    """Constrói a lista determinística de (texto, label).

    Para cada classe, gera ROWS_PER_CLASS combinações de sujeito,
    condição e contexto, embaralhadas com seed fixo.

    Returns:
        Lista de tuplas (texto do laudo, nível de urgência).
    """
    randomizer = random.Random(SEED)
    rows: list[tuple[str, int]] = []
    for label in URGENCY_MAP:
        conditions = CONDITIONS[label]
        contexts = CONTEXTS[label]
        for index in range(ROWS_PER_CLASS):
            subject = SUBJECTS[index % len(SUBJECTS)]
            condition = conditions[index % len(conditions)]
            context = contexts[(index + label) % len(contexts)]
            text = f"{subject} com {condition}, {context}"
            rows.append((text, label))
    randomizer.shuffle(rows)
    return rows


def save_dataset(rows: list[tuple[str, int]]) -> None:
    """Persiste o dataset sintético em data/raw/laudos.csv.

    Args:
        rows: Lista de tuplas (texto, label) a serem gravadas.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_csv_records(rows, RAW_DATA_PATH)


def main() -> None:
    """Gera e salva o dataset sintético de laudos médicos."""
    dataset = build_dataset()
    save_dataset(dataset)
    print(f"[OK] Dataset sintético gerado: {len(dataset)} laudos em {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
