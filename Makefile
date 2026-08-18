# =============================================================================
# Tech Challenge Fase 3 - Makefile
# =============================================================================

PYTHON := uv run python

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: \
	help \
	sync setup verify \
	test lint \
	pipeline train pipeline-live train-live evaluate-live data-synthetic \
	dvc-remote \
	docker-build docker-run docker-train \
	airflow-up airflow-down airflow-logs \
	api-up api-down benchmark

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Tech Challenge Fase 3 - Comandos Disponíveis"
	@echo ""
	@echo "Setup:"
	@echo "  make verify          - Validar ambiente (scripts/validate_env.py)"
	@echo "  make sync            - Sincronizar dependências (uv sync)"
	@echo "  make setup           - Configurar ambiente (.env, deps, pre-commit)"
	@echo ""
	@echo "Qualidade:"
	@echo "  make test            - Rodar testes (pytest)"
	@echo "  make lint            - Verificar código com ruff"
	@echo ""
	@echo "Pipeline DVC:"
	@echo "  make data-synthetic - Gerar dataset sintético e registrar no DVC"
	@echo "  make pipeline        - Reexecutar pipeline DVC completo (dvc repro -v)"
	@echo "  make train           - Reexecutar stage de treino (dvc repro train -v)"
	@echo "  make pipeline-live   - Rodar pipeline completo direto (sem DVC) com logs live"
	@echo "  make train-live      - Rodar apenas o treino direto (sem DVC) com logs live"
	@echo "  make evaluate-live   - Rodar apenas a avaliação direto (sem DVC) com logs live"
	@echo "  make dvc-remote      - Configurar remote DVC OneDrive (opcional, via .env)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Construir imagem Docker da API"
	@echo "  make docker-run      - Rodar API em container Docker"
	@echo "  make docker-train    - Rodar pipeline DVC dentro do container (perfil train)"
	@echo "  make airflow-up      - Subir stack Airflow + MLflow (perfil airflow)"
	@echo "  make airflow-down    - Parar stack Airflow"
	@echo "  make airflow-logs    - Logs da stack Airflow"
	@echo ""
	@echo "Modelo e Benchmark:"
	@echo "  make api-up          - Subir API em Docker (build + health check)"
	@echo "  make api-down        - Parar API"
	@echo "  make benchmark       - Benchmark de latência (P50/P95/P99, sobe API se necessário)"
	@echo ""

# ---------------------------------------------------------------------------
# Setup e ambiente
# ---------------------------------------------------------------------------
sync:
	uv sync

setup: sync
	@echo "Configurando ambiente..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "[OK] .env criado a partir de .env.example"; fi
	@uv run pre-commit install
	@$(MAKE) dvc-remote
	@echo "Setup concluído!"

verify:
	@echo "Validando ambiente..."
	uv run python scripts/validate_env.py

# ---------------------------------------------------------------------------
# Qualidade
# ---------------------------------------------------------------------------
test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# ---------------------------------------------------------------------------
# Pipeline DVC
# ---------------------------------------------------------------------------
pipeline:
	@echo "Reexecutando pipeline DVC completo (saída live)..."
	uv run dvc repro -v

train:
	@echo "Reexecutando stage de treino do pipeline DVC (saída live)..."
	uv run dvc repro treino -v

pipeline-live:
	@echo "Rodando pipeline COMPLETO direto (sem DVC) com logs live..."
	uv run python -m src.preprocess.run
	uv run python -m src.train.run
	uv run python -m src.evaluate.run
	@echo "[OK] pipeline concluído."

train-live:
	@echo "Rodando estágio de TREINO direto (sem DVC) com logs live..."
	uv run python -m src.train.run
	@echo "[OK] treino concluído."

evaluate-live:
	@echo "Rodando estágio de AVALIAÇÃO direto (sem DVC) com logs live..."
	uv run python -m src.evaluate.run
	@echo "[OK] avaliação concluída."

dvc-remote:
	@URL=$$(grep DVC_ONEDRIVE_REMOTE_URL .env 2>/dev/null | cut -d= -f2-); \
	if [ -n "$$URL" ]; then \
		uv run dvc remote add -f onedrive "$$URL"; \
		echo "[OK] Remote DVC OneDrive configurado."; \
	else \
		echo "[SKIP] DVC_ONEDRIVE_REMOTE_URL vazio — remote opcional, pipeline roda self-contained."; \
	fi

# ---------------------------------------------------------------------------
# Dados sintéticos
# ---------------------------------------------------------------------------
data-synthetic:
	@echo "Gerando dataset sintético..."
	PYTHONPATH=. uv run python scripts/generate_synthetic_data.py
	uv run dvc add data/raw/laudos.csv
	git add data/raw/laudos.csv.dvc
	@echo "[OK] dataset sintético gerado e registrado no DVC."

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
api-up: docker-build
	@echo "Subindo API em container Docker..."
	docker compose -f docker/docker-compose.yml --env-file .env up -d api
	@echo "Aguardando API ficar saudável..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://localhost:$$(grep API_PORT .env 2>/dev/null | cut -d= -f2 || echo 8000)/health > /dev/null 2>&1; then \
			echo "[OK] API saudável em http://localhost:$$(grep API_PORT .env 2>/dev/null | cut -d= -f2 || echo 8000)"; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "[ERRO] API não ficou saudável a tempo."; exit 1

api-down:
	@echo "Parando API..."
	docker compose -f docker/docker-compose.yml --env-file .env down api

benchmark: docker-build
	@echo "Subindo API em container Docker (rate limit desabilitado para benchmark)..."
	RATE_LIMIT_MAX_PER_IP=0 docker compose -f docker/docker-compose.yml --env-file .env up -d api
	@echo "Aguardando API ficar saudável..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://localhost:$$(grep API_PORT .env 2>/dev/null | cut -d= -f2 || echo 8000)/health > /dev/null 2>&1; then \
			echo "[OK] API saudável"; \
			break; \
		fi; \
		sleep 2; \
	done
	@echo "Executando benchmark de latência..."
	uv run python scripts/benchmark.py
	docker compose -f docker/docker-compose.yml --env-file .env stop api

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:
	@echo "Construindo imagem Docker..."
	@rm -rf /tmp/triage-api-build && mkdir -p /tmp/triage-api-build
	@cp -r pyproject.toml uv.lock src /tmp/triage-api-build/
	docker build -t triage-api -f docker/Dockerfile /tmp/triage-api-build
	@rm -rf /tmp/triage-api-build
	@echo "[OK] Imagem construída. O modelo é montado via volume em docker-compose (../models:/app/models)."

docker-run: docker-build
	@echo "Rodando API em container Docker..."
	docker compose -f docker/docker-compose.yml --env-file .env up -d
	@echo "API disponível em http://localhost:$$(grep API_PORT .env 2>/dev/null | cut -d= -f2 || echo 8000)"
	@echo "Para parar: docker compose -f docker/docker-compose.yml down"

docker-train: data-synthetic
	@echo "Docker: Rodando pipeline DVC completo no container (perfil train)..."
	docker compose -f docker/docker-compose.yml --env-file .env --profile train run --rm train
	@echo "[OK] Pipeline concluído."

# ---------------------------------------------------------------------------
# Airflow
# ---------------------------------------------------------------------------
airflow-up:
	@echo "Subindo stack Airflow + MLflow..."
	docker compose -f docker/docker-compose.yml --env-file .env --profile airflow up -d --build
	@echo "Airflow UI: http://localhost:$$(grep AIRFLOW_PORT .env 2>/dev/null | cut -d= -f2 || echo 8080)"
	@echo "MLflow UI:   http://localhost:$$(grep MLFLOW_PORT .env 2>/dev/null | cut -d= -f2 || echo 5001)"

airflow-down:
	@echo "Parando stack Airflow..."
	docker compose -f docker/docker-compose.yml --profile airflow down

airflow-logs:
	docker compose -f docker/docker-compose.yml --profile airflow logs -f
