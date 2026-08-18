"""Benchmark de latência da API de triagem.

Realiza warmup + N requisições ao endpoint /predict e calcula
os percentis P50, P95 e P99 (nunca média).

Uso:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --url http://localhost:8000 --requests 200 --warmup 10
"""

import argparse
import statistics
import time

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_REQUESTS = 100
DEFAULT_WARMUP = 10
REQUEST_TIMEOUT = 30.0
HEALTH_TIMEOUT = 10.0
HTTP_OK = 200
MIN_SAMPLE_SIZE = 100

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


def _send_request(endpoint: str, text: str, timeout: float) -> tuple[float, int, str]:
    """Envia uma requisição POST e retorna latência, status e corpo.

    Args:
        endpoint: URL do endpoint /predict.
        text: Texto do laudo médico.
        timeout: Tempo limite em segundos.

    Returns:
        Tupla (latência em ms, status HTTP, corpo da resposta).
    """
    start = time.perf_counter()
    response = httpx.post(endpoint, json={"text": text}, timeout=timeout)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, response.status_code, response.text


def run_warmup(endpoint: str, warmup: int) -> None:
    """Executa requisições de warmup.

    Args:
        endpoint: URL do endpoint /predict.
        warmup: Número de requisições de warmup.
    """
    print(f"Warmup: {warmup} requisições...")
    for i in range(warmup):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        elapsed, status, body = _send_request(endpoint, text, REQUEST_TIMEOUT)
        if status != HTTP_OK:
            raise RuntimeError(f"Warmup falhou com status {status}: {body}")


def run_benchmark(endpoint: str, n: int) -> list[float]:
    """Executa N requisições de benchmark.

    Args:
        endpoint: URL do endpoint /predict.
        n: Número de requisições.

    Returns:
        Lista de latências em milissegundos.
    """
    latencies: list[float] = []
    print(f"Benchmark: {n} requisições...")
    for i in range(n):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        elapsed, status, body = _send_request(endpoint, text, REQUEST_TIMEOUT)
        if status != HTTP_OK:
            raise RuntimeError(f"Requisição {i + 1} falhou com status {status}: {body}")
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
    args = parse_args()
    _validate_sample_size(args.requests)

    print(f"Conectando em {args.url}...")
    health = httpx.get(f"{args.url}/health", timeout=HEALTH_TIMEOUT)
    if health.status_code != HTTP_OK:
        raise RuntimeError(f"API não saudável: status {health.status_code}")
    print("[OK] Health check passado")

    run_warmup(f"{args.url}/predict", args.warmup)
    benchmark_latencies = run_benchmark(f"{args.url}/predict", args.requests)

    stats = compute_percentiles(benchmark_latencies)
    print_results(stats, args.requests)


if __name__ == "__main__":
    main()
