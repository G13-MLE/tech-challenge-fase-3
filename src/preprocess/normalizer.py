"""Estratégias de normalização de texto para laudos médicos."""

import re
import unicodedata
from abc import ABC, abstractmethod

__all__ = [
    "TextNormalizer",
    "LowercaseNormalizer",
    "PunctuationNormalizer",
    "ComposedNormalizer",
    "compose_normalizers",
]


class TextNormalizer(ABC):
    """Interface abstrata para estratégias de normalização de texto."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normaliza o texto do laudo médico.

        Args:
            text: Texto bruto do laudo.

        Returns:
            Texto normalizado.
        """


class LowercaseNormalizer(TextNormalizer):
    """Converte o texto para minúsculas e remove acentuação."""

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        stripped = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        return stripped.lower()


class PunctuationNormalizer(TextNormalizer):
    """Remove pontuação e espaços redundantes do texto.

    Nota: preserva dígitos (relevantes para laudos como "tipo 2")
    e substitui underscores por espaços.
    """

    def normalize(self, text: str) -> str:
        without_punctuation = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        without_underscores = without_punctuation.replace("_", " ")
        return re.sub(r"\s+", " ", without_underscores).strip()


class ComposedNormalizer(TextNormalizer):
    """Aplica normalizadores em sequência."""

    def __init__(self, normalizers: tuple[TextNormalizer, ...]) -> None:
        self._normalizers = normalizers

    def normalize(self, text: str) -> str:
        result = text
        for normalizer in self._normalizers:
            result = normalizer.normalize(result)
        return result


def compose_normalizers(*normalizers: TextNormalizer) -> TextNormalizer:
    """Compõe múltiplas estratégias de normalização em uma única.

    Args:
        *normalizers: Estratégias a serem aplicadas em sequência.

    Returns:
        Estratégia composta que aplica os normalizadores na ordem dada.
    """
    return ComposedNormalizer(normalizers)
