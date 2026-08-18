"""Enumeração dos níveis de urgência para classificação de laudos médicos."""

from enum import Enum

__all__ = ["UrgencyLevel", "URGENCY_MAP"]


class UrgencyLevel(str, Enum):
    """Níveis de urgência para triagem de laudos médicos.

    Herda de str para serialização direta em JSON via Pydantic/FastAPI.
    """

    NORMAL = "normal"
    ATENCAO = "atenção"
    URGENTE = "urgente"


URGENCY_MAP: dict[int, UrgencyLevel] = {
    0: UrgencyLevel.NORMAL,
    1: UrgencyLevel.ATENCAO,
    2: UrgencyLevel.URGENTE,
}
