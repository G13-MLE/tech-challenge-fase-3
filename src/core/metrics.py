"""Métricas Prometheus da API de triagem.

Expõe contadores e histogramas para observabilidade via endpoint /metrics.
"""

from prometheus_client import Counter, Histogram

__all__ = [
    "app_requests_total",
    "app_request_latency_seconds",
    "predictions_total",
    "prediction_latency_seconds",
]

app_requests_total = Counter(
    "app_requests_total",
    "Total de requisições HTTP por método, endpoint e status",
    ["method", "endpoint", "http_status"],
)

app_request_latency_seconds = Histogram(
    "app_request_latency_seconds",
    "Latência das requisições HTTP em segundos",
    ["method", "endpoint", "http_status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

predictions_total = Counter(
    "predictions_total",
    "Total de predições por nível de urgência",
    ["urgency"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Latência das predições em segundos",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
