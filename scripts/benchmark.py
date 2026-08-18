"""Benchmark de latência da API de triagem.

Realiza warmup + N requisições ao endpoint /api/v1/predict e calcula
os percentis P50, P95 e P99 (nunca média).

Uso:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --url http://localhost:8000 --requests 200 --warmup 10
    uv run python scripts/benchmark.py --concurrency 8 --requests 500
    uv run python scripts/benchmark.py --api-key dev-api-key
"""

import argparse
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_PREDICT_PATH = "/api/v1/predict"
DEFAULT_REQUESTS = 100
DEFAULT_WARMUP = 10
DEFAULT_CONCURRENCY = 1
REQUEST_TIMEOUT = 30.0
HEALTH_TIMEOUT = 10.0
HTTP_OK = 200
HTTP_TOO_MANY_REQUESTS = 429
MIN_SAMPLE_SIZE = 100
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_INITIAL_BACKOFF = 1.0

logger = logging.getLogger(__name__)

SAMPLE_TEXTS = [
    "paciente com pneumonia bacteriana grave",
    "diabetes tipo 2 descompensada com glicemia elevada",
    "asma crônica com sibilos e dispneia",
    "hipertensão arterial com pressão acima de 140x90",
    "hérnia inguinal redutível sem complicação",
    "crise asmática com broncoespasmo",
    "pneumonia comunitária com infiltrado bilateral",
    "diabetes mellitus com neuropatia periférica",
]


def _send_request(
    endpoint: str,
    text: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[float, int, str]:
    """Envia uma requisição POST e retorna latência, status e corpo.

    Em caso de rate limit (429), realiza retry com backoff exponencial.

    Args:
        endpoint: URL do endpoint /predict.
        text: Texto do laudo médico.
        timeout: Tempo limite em segundos.
        headers: Headers HTTP adicionais (ex.: X-API-Key).

    Returns:
        Tupla (latência em ms, status HTTP, corpo da resposta).
    """
    request_headers = headers or {}
    elapsed = 0.0
    response = None  # will be assigned in the loop below
    for attempt in range(RATE_LIMIT_MAX_RETRIES):
        start = time.perf_counter()
        response = httpx.post(
            endpoint, json={"text": text}, headers=request_headers, timeout=timeout
        )
        elapsed = (time.perf_counter() - start) * 1000
        if response.status_code == HTTP_TOO_MANY_REQUESTS:
            backoff = RATE_LIMIT_INITIAL_BACKOFF * (2**attempt)
            logger.warning(
                "Rate limit atingido (429), retry em %.1fs (tentativa %d/%d)",
                backoff,
                attempt + 1,
                RATE_LIMIT_MAX_RETRIES,
            )
            time.sleep(backoff)
            continue
        return elapsed, response.status_code, response.text
    # All retries exhausted — response was set in the last loop iteration
    assert response is not None, "Unexpected: no HTTP response after retries"
    return elapsed, response.status_code, response.text


def run_warmup(endpoint: str, warmup: int, headers: dict[str, str] | None = None) -> None:
    """Executa requisições de warmup.

    Args:
        endpoint: URL do endpoint /predict.
        warmup: Número de requisições de warmup.
        headers: Headers HTTP adicionais (ex.: X-API-Key).
    """
    print(f"Warmup: {warmup} requisições...")
    for i in range(warmup):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        elapsed, status, body = _send_request(endpoint, text, REQUEST_TIMEOUT, headers=headers)
        if status != HTTP_OK:
            raise RuntimeError(f"Warmup falhou com status {status}: {body}")


def run_benchmark(
    endpoint: str,
    n: int,
    concurrency: int = 1,
    headers: dict[str, str] | None = None,
) -> list[float]:
    """Executa N requisições de benchmark, opcionalmente em paralelo.

    Args:
        endpoint: URL do endpoint /predict.
        n: Número de requisições.
        concurrency: Número de requisições simultâneas (1 = sequencial).
        headers: Headers HTTP adicionais (ex.: X-API-Key).

    Returns:
        Lista de latências em milissegundos.
    """
    latencies: list[float] = []
    print(f"Benchmark: {n} requisições (concorrência={concurrency})...")
    if concurrency <= 1:
        for i in range(n):
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            elapsed, status, body = _send_request(endpoint, text, REQUEST_TIMEOUT, headers=headers)
            if status != HTTP_OK:
                raise RuntimeError(f"Requisição {i + 1} falhou com status {status}: {body}")
            latencies.append(elapsed)
        return latencies
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _send_request,
                endpoint,
                SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)],
                REQUEST_TIMEOUT,
                headers,
            )
            for i in range(n)
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            elapsed, status, body = future.result()
            if status != HTTP_OK:
                raise RuntimeError(f"Requisição {i} falhou com status {status}: {body}")
            latencies.append(elapsed)
    return latencies


def compute_percentiles(latencies: list[float]) -> dict[str, float]:
    """Calcula os percentis P50, P95 e P99 das latências.

    Args:
        latencies: Lista de latências em milissegundos.

    Returns:
        Dicionário com P50, P95 e P99.
    """
    sorted_lat = sorted(latencies)
    return {
        "P50": statistics.quantiles(sorted_lat, n=2)[0],
        "P95": statistics.quantiles(sorted_lat, n=20)[18],
        "P99": statistics.quantiles(sorted_lat, n=100)[98],
        "min": sorted_lat[0],
        "max": sorted_lat[-1],
    }


def parse_args() -> argparse.Namespace:
    """Parseia os argumentos da linha de comando.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Benchmark de latência da API")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key para autenticação (header X-API-Key)",
    )
    return parser.parse_args()


def print_results(stats: dict[str, float], n_requests: int) -> None:
    """Imprime os resultados do benchmark.

    Args:
        stats: Dicionário com P50, P95, P99, min e max.
        n_requests: Número de requisições executadas.
    """
    print("\n" + "=" * 50)
    print(f"Benchmark Results ({n_requests} requisições)")
    print("=" * 50)
    print(f"  P50 : {stats['P50']:.2f} ms")
    print(f"  P95 : {stats['P95']:.2f} ms")
    print(f"  P99 : {stats['P99']:.2f} ms")
    print(f"  min : {stats['min']:.2f} ms")
    print(f"  max : {stats['max']:.2f} ms")
    print("=" * 50)


def _validate_sample_size(n: int) -> None:
    """Valida se o número de requisições é suficiente para percentis.

    Args:
        n: Número de requisições.

    Raises:
        ValueError: Se n for menor que MIN_SAMPLE_SIZE.
    """
    if n < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"Número de requisições ({n}) insuficiente para percentis "
            f"estatisticamente significativos. Mínimo: {MIN_SAMPLE_SIZE}."
        )


def main() -> None:
    """Ponto de entrada do benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()
    _validate_sample_size(args.requests)

    # Build request headers if API key is provided
    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    print(f"Conectando em {args.url}...")
    try:
        health = httpx.get(f"{args.url}/health", timeout=HEALTH_TIMEOUT)
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"Não foi possível conectar em {args.url}. "
            f"Verifique se o servidor está rodando (ex.: make api-up). "
            f"Erro original: {exc}"
        ) from exc
    if health.status_code != HTTP_OK:
        raise RuntimeError(f"API não saudável: status {health.status_code}")
    print("[OK] Health check passado")

    predict_endpoint = f"{args.url}{DEFAULT_PREDICT_PATH}"
    run_warmup(predict_endpoint, args.warmup, headers=headers)
    benchmark_latencies = run_benchmark(
        predict_endpoint, args.requests, args.concurrency, headers=headers
    )

    stats = compute_percentiles(benchmark_latencies)
    print_results(stats, args.requests)


if __name__ == "__main__":
    main()
