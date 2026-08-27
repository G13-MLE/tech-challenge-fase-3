"""Benchmark de latência da API de triagem.

Realiza warmup + N requisições ao endpoint /api/v1/predict e calcula
os percentis P50, P95 e P99 (nunca média).

Suporta modo de comparação entre backends joblib e ONNX, com relatório
de tamanho de artefato e tabela comparativa em Markdown.

Uso:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --url http://localhost:8000 --requests 200 --warmup 10
    uv run python scripts/benchmark.py --concurrency 8 --requests 500
    uv run python scripts/benchmark.py --api-key dev-api-key
    uv run python scripts/benchmark.py --compare --output reports/benchmark_comparison.md
"""

import argparse
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
MODEL_DIR = Path("models")
JOBLIB_FILE = MODEL_DIR / "model.joblib"
ONNX_FILE = MODEL_DIR / "model.onnx"

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
    response = None
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
        Dicionário com P50, P95, P99, min e max.
    """
    sorted_lat = sorted(latencies)
    return {
        "P50": statistics.quantiles(sorted_lat, n=2)[0],
        "P95": statistics.quantiles(sorted_lat, n=20)[18],
        "P99": statistics.quantiles(sorted_lat, n=100)[98],
        "min": sorted_lat[0],
        "max": sorted_lat[-1],
    }


def get_artifact_size(path: Path) -> int | None:
    """Retorna o tamanho de um arquivo de artefato em bytes, ou None se não existir.

    Args:
        path: Caminho do arquivo.

    Returns:
        Tamanho em bytes ou None.
    """
    if path.exists():
        return path.stat().st_size
    return None


def format_size(size_bytes: int) -> str:
    """Formata bytes em string legível (KB ou MB).

    Args:
        size_bytes: Tamanho em bytes.

    Returns:
        String formatada.
    """
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.2f} MB"
    return f"{size_bytes / 1_000:.1f} KB"


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


def print_comparison(
    joblib_stats: dict[str, float],
    onnx_stats: dict[str, float],
    n_requests: int,
) -> None:
    """Imprime tabela comparativa entre backends joblib e ONNX.

    Args:
        joblib_stats: Estatísticas do backend joblib.
        onnx_stats: Estatísticas do backend ONNX.
        n_requests: Número de requisições por backend.
    """
    joblib_size = get_artifact_size(JOBLIB_FILE)
    onnx_size = get_artifact_size(ONNX_FILE)

    print("\n" + "=" * 60)
    print(f"Comparação de Latência: joblib vs ONNX ({n_requests} req/backend)")
    print("=" * 60)
    print(f"{'Métrica':<10} {'joblib':>12} {'ONNX':>12} {'Δ':>12}")
    print("-" * 60)
    for metric in ["P50", "P95", "P99", "min", "max"]:
        j = joblib_stats[metric]
        o = onnx_stats[metric]
        delta = o - j
        pct = (delta / j * 100) if j != 0 else 0
        print(f"{metric:<10} {j:>10.2f} ms {o:>10.2f} ms {delta:>+9.2f} ms ({pct:+.1f}%)")
    print("-" * 60)
    if joblib_size is not None:
        print(f"{'Artefato':<10} {format_size(joblib_size):>12}", end="")
    else:
        print(f"{'Artefato':<10} {'N/A':>12}", end="")
    if onnx_size is not None:
        print(f" {format_size(onnx_size):>12}")
    else:
        print(f" {'N/A':>12}")
    print("=" * 60)


def write_comparison_markdown(
    joblib_stats: dict[str, float],
    onnx_stats: dict[str, float],
    n_requests: int,
    output_path: Path,
) -> None:
    """Escreve tabela comparativa em formato Markdown.

    Args:
        joblib_stats: Estatísticas do backend joblib.
        onnx_stats: Estatísticas do backend ONNX.
        n_requests: Número de requisições por backend.
        output_path: Caminho do arquivo Markdown de saída.
    """
    joblib_size = get_artifact_size(JOBLIB_FILE)
    onnx_size = get_artifact_size(ONNX_FILE)

    lines = [
        "## Comparação de Latência: joblib vs ONNX",
        "",
        f"Backend: **{n_requests} requisições** cada | Warmup: {DEFAULT_WARMUP} req",
        "",
        "| Métrica | joblib | ONNX | Δ |",
        "|---------|-------|------|---|",
    ]
    for metric in ["P50", "P95", "P99", "min", "max"]:
        j = joblib_stats[metric]
        o = onnx_stats[metric]
        delta = o - j
        pct = (delta / j * 100) if j != 0 else 0
        lines.append(f"| {metric} | {j:.2f} ms | {o:.2f} ms | {delta:+.2f} ms ({pct:+.1f}%) |")

    lines.append("")
    if joblib_size is not None and onnx_size is not None:
        lines.append(f"| Artefato | {format_size(joblib_size)} | {format_size(onnx_size)} | — |")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Tabela comparativa salva em %s", output_path)


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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Modo comparação: roda benchmark com joblib e ONNX e exibe tabela comparativa",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Caminho do arquivo Markdown para salvar a tabela comparativa (modo --compare)",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()
    _validate_sample_size(args.requests)

    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    if args.compare:
        run_comparison_benchmark(args, headers)
    else:
        run_single_benchmark(args, headers)


def run_single_benchmark(args: argparse.Namespace, headers: dict[str, str]) -> None:
    """Executa benchmark com um único backend (backend atual da API).

    Args:
        args: Argumentos da linha de comando.
        headers: Headers HTTP.
    """
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


def run_comparison_benchmark(args: argparse.Namespace, headers: dict[str, str]) -> None:
    """Executa benchmark comparativo entre backends joblib e ONNX.

    Requer que a API esteja rodando com Docker Compose. Para cada backend,
    reinicia o container com a variável MODEL_BACKEND correspondente,
    aguarda health check e executa o benchmark.

    Args:
        args: Argumentos da linha de comando.
        headers: Headers HTTP.
    """
    import subprocess

    results: dict[str, dict[str, float]] = {}
    compose_cmd = ["docker", "compose", "-f", "docker/docker-compose.yml"]

    for backend in ["joblib", "onnx"]:
        print(f"\n{'=' * 60}")
        print(f"  Benchmark com backend: {backend}")
        print(f"{'=' * 60}")

        env = os.environ.copy()
        env["MODEL_BACKEND"] = backend

        print(f"Reiniciando API com MODEL_BACKEND={backend}...")
        subprocess.run(
            compose_cmd + ["--env-file", ".env", "stop", "api"],
            check=False,
            capture_output=True,
        )

        docker_env = env.copy()
        subprocess.run(
            compose_cmd + ["--env-file", ".env", "up", "-d", "api"],
            env=docker_env,
            check=True,
            capture_output=True,
        )

        print("Aguardando API ficar saudável...")
        for _ in range(30):
            try:
                resp = httpx.get(f"{args.url}/health", timeout=HEALTH_TIMEOUT)
                if resp.status_code == HTTP_OK:
                    print(f"[OK] API saudável com backend {backend}")
                    break
            except httpx.ConnectError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"API não ficou saudável com backend {backend}")

        predict_endpoint = f"{args.url}{DEFAULT_PREDICT_PATH}"
        run_warmup(predict_endpoint, args.warmup, headers=headers)
        latencies = run_benchmark(
            predict_endpoint, args.requests, args.concurrency, headers=headers
        )
        results[backend] = compute_percentiles(latencies)

    print_comparison(results["joblib"], results["onnx"], args.requests)

    if args.output:
        write_comparison_markdown(
            results["joblib"], results["onnx"], args.requests, Path(args.output)
        )

    subprocess.run(
        compose_cmd + ["--env-file", ".env", "stop", "api"],
        check=False,
        capture_output=True,
    )


if __name__ == "__main__":
    main()
