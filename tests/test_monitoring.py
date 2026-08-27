"""Testes smoke da stack de monitoramento (Prometheus + Grafana)."""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
DOCKER_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DASHBOARD_JSON = CONFIGS / "grafana" / "dashboards" / "triage-api.json"

EXPECTED_PANELS = {
    "Taxa de Requisições": "sum(rate(app_requests_total[5m]))",
    "Latência P95": (
        "histogram_quantile(0.95, sum by (le) (rate(app_request_latency_seconds_bucket[5m])))"
    ),
    "Taxa de Erro": (
        'sum(rate(app_requests_total{http_status=~"[45].."}[5m]))'
        " / sum(rate(app_requests_total[5m]))"
    ),
}


def test_dashboard_json_parses() -> None:
    """Dashboard JSON deve ser válido e conter 3 painéis."""
    data = json.loads(DASHBOARD_JSON.read_text())
    assert data["title"] == "Triage API"
    panels = data["panels"]
    assert len(panels) == 3, f"esperado 3 painéis, encontrado {len(panels)}"


def test_dashboard_contains_expected_panels() -> None:
    """Cada painel deve ter o título e a expressão PromQL esperados."""
    data = json.loads(DASHBOARD_JSON.read_text())
    found = {p["title"]: p["targets"][0]["expr"] for p in data["panels"]}
    for title, expr in EXPECTED_PANELS.items():
        assert title in found, f"painel '{title}' ausente"
        assert found[title] == expr, f"expressão inesperada para '{title}': {found[title]}"


def test_dashboard_uses_prometheus_datasource() -> None:
    """Os painéis devem referenciar o datasource Prometheus com uid explícito."""
    data = json.loads(DASHBOARD_JSON.read_text())
    for p in data["panels"]:
        ds = p.get("datasource", {})
        assert ds.get("type") == "prometheus", f"painel '{p['title']}' sem datasource Prometheus"
        assert ds.get("uid") == "prometheus", f"painel '{p['title']}' uid diferente de 'prometheus'"


def test_compose_has_grafana_service() -> None:
    """docker-compose deve declarar o serviço grafana no perfil monitoring."""
    compose = yaml.safe_load(DOCKER_COMPOSE.read_text())
    services = compose["services"]
    assert "grafana" in services, "serviço 'grafana' ausente no docker-compose"
    grafana = services["grafana"]
    assert "monitoring" in grafana["profiles"], "grafana deve estar no perfil 'monitoring'"
    ports = grafana["ports"]
    assert any("3000" in p for p in ports), "grafana deve expor a porta 3000"


def test_compose_grafana_depends_on_prometheus() -> None:
    """grafana deve depender do prometheus (service_healthy)."""
    compose = yaml.safe_load(DOCKER_COMPOSE.read_text())
    grafana = compose["services"]["grafana"]
    deps = grafana.get("depends_on", {})
    assert "prometheus" in deps, "grafana deve depender de prometheus"
    assert deps["prometheus"]["condition"] == "service_healthy"


def test_compose_grafana_mounts_provisioning() -> None:
    """grafana deve montar diretórios de provisioning e dashboards."""
    compose = yaml.safe_load(DOCKER_COMPOSE.read_text())
    grafana = compose["services"]["grafana"]
    volumes = grafana["volumes"]
    assert any("provisioning" in v and "/etc/grafana/provisioning" in v for v in volumes)
    assert any("dashboards" in v and "/var/lib/grafana/dashboards" in v for v in volumes)
