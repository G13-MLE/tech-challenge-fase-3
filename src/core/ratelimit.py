"""Rate limiting simples em memória por IP.

Implementa um token bucket por IP de entrada (in-memory, sem dependências
externas). Adequado para desenvolvimento e deploys single-instance.
"""

import threading
import time
from collections import defaultdict

__all__ = ["RateLimiter"]


class RateLimiter:
    """Token bucket em memória por endereço IP.

    Attributes:
        max_requests: Número máximo de requisições permitidas.
        window_seconds: Janela de tempo (em segundos) para o limite.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Verifica se a requisição do cliente é permitida.

        Args:
            key: Identificador do cliente (ex.: IP).

        Returns:
            True se permitido, False se excedeu o limite.
            Sempre retorna True se max_requests for 0 (sem limite).
        """
        if self._max_requests <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired(now)
            window_start = now - self._window_seconds
            recent = [ts for ts in self._hits[key] if ts >= window_start]
            if len(recent) >= self._max_requests:
                self._hits[key] = recent
                return False
            recent.append(now)
            self._hits[key] = recent
            return True

    def _cleanup_expired(self, now: float) -> None:
        """Remove entradas expiradas de todos os IPs.

        Args:
            now: Timestamp atual (time.monotonic).
        """
        window_start = now - self._window_seconds
        expired_keys = [k for k, v in self._hits.items() if not v or v[-1] < window_start]
        for k in expired_keys:
            del self._hits[k]
