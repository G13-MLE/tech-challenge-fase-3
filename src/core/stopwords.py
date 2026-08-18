"""Stopwords em português para o TF-IDF.

Lista minimalista de conectivos e artigos mais frequentes em laudos
médicos em português. Pode ser substituída por uma lista completa via
`configs/params.yaml` (`tfidf_stopwords`).
"""

__all__ = ["PORTUGUESE_STOPWORDS"]

PORTUGUESE_STOPWORDS: list[str] = [
    "de",
    "a",
    "o",
    "que",
    "e",
    "do",
    "da",
    "em",
    "um",
    "para",
    "com",
    "nao",
    "uma",
    "os",
    "no",
    "se",
    "na",
    "por",
    "mais",
    "as",
    "dos",
    "como",
    "mas",
    "foi",
    "ao",
    "ele",
    "das",
    "seu",
    "sua",
    "ou",
    "ser",
    "muito",
    "ha",
    "num",
    "numa",
]
