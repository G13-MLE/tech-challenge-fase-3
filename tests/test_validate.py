"""Testes da validação de dados brutos."""

import pytest

from src.validate.run import validate_raw_data


def write_csv(path, header: list[str], rows: list[tuple[str, ...]]) -> None:
    """Escreve um CSV com header e linhas."""
    lines = [",".join(header)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestValidateRawData:
    """Testes da validação dos dados brutos."""

    def test_valid_data(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_csv(path, ["text", "label"], [("paciente com asma", "0"), ("pneumonia", "2")])
        texts, labels = validate_raw_data(path)
        assert texts == ["paciente com asma", "pneumonia"]
        assert labels == [0, 2]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_raw_data(tmp_path / "inexistente.csv")

    def test_missing_text_column_raises(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_csv(path, ["label"], [("0",)])
        with pytest.raises(ValueError, match="'text' ausente"):
            validate_raw_data(path)

    def test_missing_label_column_raises(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_csv(path, ["text"], [("paciente",)])
        with pytest.raises(ValueError, match="'label' ausente"):
            validate_raw_data(path)

    def test_empty_text_raises(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_csv(path, ["text", "label"], [("  ", "0")])
        with pytest.raises(ValueError, match="Texto vazio"):
            validate_raw_data(path)

    def test_invalid_label_raises(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_csv(path, ["text", "label"], [("paciente", "5")])
        with pytest.raises(ValueError, match="fora do esperado"):
            validate_raw_data(path)

    def test_non_numeric_label_raises(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_csv(path, ["text", "label"], [("paciente", "urgente")])
        with pytest.raises(ValueError, match="Rótulo inválido"):
            validate_raw_data(path)

    def test_empty_dataset_raises(self, tmp_path) -> None:
        path = tmp_path / "laudos.csv"
        write_csv(path, ["text", "label"], [])
        with pytest.raises(ValueError, match="nenhum registro"):
            validate_raw_data(path)
