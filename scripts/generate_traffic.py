"""Gerador de tráfego para popular métricas Prometheus da API de triagem.

Envia requisições variadas (predict, batch, health, erros intencionais)
para que o Prometheus colete dados reais nos painéis.

Uso:
    uv run python scripts/generate_traffic.py
    uv run python scripts/generate_traffic.py --url http://localhost:8000 --duration 60 --rate 5
    uv run python scripts/generate_traffic.py --api-key dev-api-key
"""

import argparse
import logging
import random
import time

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_DURATION = 30
DEFAULT_RATE = 5
REQUEST_TIMEOUT = 10.0
HEALTH_TIMEOUT = 5.0

logger = logging.getLogger(__name__)

PREDICT_TEXTS = [
    "paciente com pneumonia bacteriana grave",
    "diabetes tipo 2 descompensada com glicemia elevada",
    "asma crônica com sibilos e dispneia",
    "hipertensão arterial com pressão acima de 140x90",
    "hérnia inguinal redutível sem complicação",
    "crise asmática com broncoespasmo",
    "pneumonia comunitária com infiltrado bilateral",
    "diabetes mellitus com neuropatia periférica",
]

BATCH_TEXTS = [
    ["paciente com pneumonia grave", "asma brônquica"],
    ["hipertensão arterial", "diabetes tipo 2"],
    ["crise hipertensiva com cefaleia", "hérnia umbilical"],
]

INTENTIONAL_ERROR_TEXTS = [
    "",
    "!!!",
]


def send_predict(client: httpx.Client, url: str, headers: dict[str, str]) -> int:
    """Envia uma requisição POST /predict e retorna o status code."""
    text = random.choice(PREDICT_TEXTS)
    try:
        response = client.post(f"{url}/api/v1/predict", json={"text": text}, headers=headers)
        return response.status_code
    except httpx.ConnectError:
        logger.warning("Conexão recusada em /predict")
        return 0


def send_batch(client: httpx.Client, url: str, headers: dict[str, str]) -> int:
    """Envia uma requisição POST /predict/batch e retorna o status code."""
    texts = random.choice(BATCH_TEXTS)
    try:
        response = client.post(
            f"{url}/api/v1/predict/batch", json={"texts": texts}, headers=headers
        )
        return response.status_code
    except httpx.ConnectError:
        logger.warning("Conexão recusada em /predict/batch")
        return 0


def send_health(client: httpx.Client, url: str) -> int:
    """Envia uma requisição GET /health e retorna o status code."""
    try:
        response = client.get(f"{url}/health")
        return response.status_code
    except httpx.ConnectError:
        logger.warning("Conexão recusada em /health")
        return 0


def send_intentional_error(client: httpx.Client, url: str, headers: dict[str, str]) -> int:
    """Envia uma requisição inválida (422) e retorna o status code."""
    text = random.choice(INTENTIONAL_ERROR_TEXTS)
    try:
        response = client.post(f"{url}/api/v1/predict", json={"text": text}, headers=headers)
        return response.status_code
    except httpx.ConnectError:
        logger.warning("Conexão recusada em erro intencional")
        return 0


def send_nonexistent_path(client: httpx.Client, url: str) -> int:
    """Envia uma requisição GET para path inexistente (404)."""
    try:
        response = client.get(f"{url}/nonexistent")
        return response.status_code
    except httpx.ConnectError:
        logger.warning("Conexão recusada em path inexistente")
        return 0


def parse_args() -> argparse.Namespace:
    """Parseia os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Gerador de tráfego para métricas Prometheus")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL base da API")
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_DURATION, help="Duração em segundos"
    )
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE, help="Requisições por segundo")
    parser.add_argument("--api-key", default=None, help="API key (header X-API-Key)")
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do gerador de tráfego."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()
    url = args.url
    duration = args.duration
    rate = args.rate

    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    print(f"Conectando em {url}...")
    try:
        health = httpx.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"Não foi possível conectar em {url}. "
            f"Verifique se o servidor está rodando (ex.: make api-up). "
            f"Erro original: {exc}"
        ) from exc
    if health.status_code != 200:
        raise RuntimeError(f"API não saudável: status {health.status_code}")
    print("[OK] Health check passado")

    request_types = [
        ("predict", 0.45),
        ("batch", 0.20),
        ("health", 0.15),
        ("error_422", 0.10),
        ("error_404", 0.10),
    ]
    weights = [w for _, w in request_types]
    names = [n for n, _ in request_types]

    total_sent = 0
    total_errors = 0
    interval = 1.0 / rate
    end_time = time.monotonic() + duration

    print(f"Gerando tráfego por {duration}s a ~{rate} req/s...")
    print(f"Tipos: {dict(zip(names, weights, strict=True))}")

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        while time.monotonic() < end_time:
            choice = random.choices(names, weights=weights, k=1)[0]

            if choice == "predict":
                status = send_predict(client, url, headers)
            elif choice == "batch":
                status = send_batch(client, url, headers)
            elif choice == "health":
                status = send_health(client, url)
            elif choice == "error_422":
                status = send_intentional_error(client, url, headers)
            elif choice == "error_404":
                status = send_nonexistent_path(client, url)
            else:
                status = 0

            total_sent += 1
            if status >= 400 or status == 0:
                total_errors += 1

            if total_sent % 50 == 0:
                logger.info("Progresso: %d req, %d erros", total_sent, total_errors)

            time.sleep(interval)

    print("\n" + "=" * 50)
    print(f"Tráfego concluído: {total_sent} requisições em {duration}s")
    print(f"Erros (4xx/5xx/falhas): {total_errors}")
    print(f"Taxa de sucesso: {(total_sent - total_errors) / total_sent * 100:.1f}%")
    print("=" * 50)
    print("\nMétricas disponíveis em:")
    print(f"  API /metrics:  {url}/metrics")
    print("  Prometheus:    http://localhost:9090")


if __name__ == "__main__":
    main()
