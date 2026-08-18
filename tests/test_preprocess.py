"""Testes do pipeline de pré-processamento."""

import csv

import pytest

from src.core.dataset import LABEL_COLUMN, TEXT_COLUMN, load_csv_records, save_csv_records
from src.preprocess.normalizer import (
    ComposedNormalizer,
    LowercaseNormalizer,
    PunctuationNormalizer,
    compose_normalizers,
)
from src.preprocess.run import (
    build_default_normalizer,
    load_raw_records,
    preprocess_text,
)


class TestNormalizers:
    """Testes das estratégias de normalização."""

    def test_lowercase_normalizer(self) -> None:
        normalizer = LowercaseNormalizer()
        assert normalizer.normalize("PACIENTE com ASMA") == "paciente com asma"

    def test_lowercase_normalizer_removes_accents(self) -> None:
        normalizer = LowercaseNormalizer()
        assert normalizer.normalize("ATENÇÃO Médica") == "atencao medica"

    def test_punctuation_normalizer(self) -> None:
        normalizer = PunctuationNormalizer()
        assert (
            normalizer.normalize("paciente, com (pneumonia) grave!")
            == "paciente com pneumonia grave"
        )

    def test_punctuation_normalizer_collapses_spaces(self) -> None:
        normalizer = PunctuationNormalizer()
        assert normalizer.normalize("paciente  com  asma") == "paciente com asma"

    def test_punctuation_normalizer_removes_underscores(self) -> None:
        normalizer = PunctuationNormalizer()
        assert normalizer.normalize("paciente_com_asma") == "paciente com asma"

    def test_composed_normalizer(self) -> None:
        normalizer = compose_normalizers(LowercaseNormalizer(), PunctuationNormalizer())
        assert normalizer.normalize("PACIENTE, com PNEUMONIA!") == "paciente com pneumonia"

    def test_composed_normalizer_is_instance(self) -> None:
        normalizer = compose_normalizers(LowercaseNormalizer(), PunctuationNormalizer())
        assert isinstance(normalizer, ComposedNormalizer)

    def test_default_normalizer_full_pipeline(self) -> None:
        normalizer = build_default_normalizer()
        result = normalizer.normalize("Paciente, com DIABETES tipo 2 (descompensada).")
        assert result == "paciente com diabetes tipo 2 descompensada"

    def test_empty_string(self) -> None:
        normalizer = build_default_normalizer()
        assert normalizer.normalize("") == ""

    def test_numbers_preserved(self) -> None:
        normalizer = build_default_normalizer()
        assert "tipo 2" in normalizer.normalize("Paciente com diabetes tipo 2")


class TestPreprocessPipeline:
    """Testes do fluxo de pré-processamento."""

    def test_load_raw_records(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_raw_csv(path, [("Paciente com pneumonia", 2), ("Asma crônica", 0)])
        records = load_raw_records(path)
        assert records == [("Paciente com pneumonia", 2), ("Asma crônica", 0)]

    def test_load_raw_records_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_raw_records(tmp_path / "inexistente.csv")

    def test_preprocess_text(self) -> None:
        normalizer = build_default_normalizer()
        assert preprocess_text("PACIENTE com PNEUMONIA!", normalizer) == "paciente com pneumonia"

    def test_save_and_load_csv_records_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "processed.csv"
        records = [("paciente com asma", 0), ("diabetes descompensada", 1)]
        save_csv_records(records, path)
        texts, labels = load_csv_records(path)
        assert texts == ["paciente com asma", "diabetes descompensada"]
        assert labels == [0, 1]

    def test_load_csv_records_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv_records(tmp_path / "inexistente.csv")


def write_raw_csv(path, rows: list[tuple[str, int]]) -> None:
    """Escreve um CSV bruto de laudos para os testes."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([TEXT_COLUMN, LABEL_COLUMN])
        writer.writerows(rows)
