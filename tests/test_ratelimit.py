"""Testes do rate limiter em memória."""

from src.core.ratelimit import RateLimiter


class TestRateLimiter:
    """Testes do token bucket por IP."""

    def test_allows_within_limit(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("ip-1")
        assert limiter.is_allowed("ip-1")
        assert limiter.is_allowed("ip-1")

    def test_blocks_after_limit(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("ip-1")
        assert limiter.is_allowed("ip-1")
        assert not limiter.is_allowed("ip-1")

    def test_ips_are_isolated(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("ip-1")
        assert not limiter.is_allowed("ip-1")
        assert limiter.is_allowed("ip-2")

    def test_window_expiry(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed("ip-1")
        assert not limiter.is_allowed("ip-1")
        import time

        time.sleep(1.1)
        assert limiter.is_allowed("ip-1")

    def test_expired_ips_are_cleaned_up(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        limiter.is_allowed("ip-1")
        limiter.is_allowed("ip-2")
        limiter.is_allowed("ip-3")
        assert len(limiter._hits) == 3
        import time

        time.sleep(1.1)
        limiter.is_allowed("ip-4")
        assert "ip-1" not in limiter._hits
        assert "ip-2" not in limiter._hits
        assert "ip-3" not in limiter._hits
