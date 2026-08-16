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
	pipeline train pipeline-live train-live \
	docker-train

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
	@echo "  make pipeline        - Reexecutar pipeline DVC completo (dvc repro -v)"
	@echo "  make train           - Reexecutar stage de treino (dvc repro train -v)"
	@echo "  make pipeline-live   - Rodar pipeline completo direto (sem DVC) com logs live"
	@echo "  make train-live      - Rodar apenas o treino direto (sem DVC) com logs live"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-train    - Rodar pipeline DVC dentro do container (perfil train)"
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

# ---------------------------------------------------------------------------
# Pipeline DVC
# ---------------------------------------------------------------------------
pipeline:
	@echo "Reexecutando pipeline DVC completo (saída live)..."
	uv run dvc repro -v

train:
	@echo "Reexecutando stage de treino do pipeline DVC (saída live)..."
	uv run dvc repro train -v

pipeline-live:
	@echo "Rodando pipeline COMPLETO direto (sem DVC) com logs live..."
	uv run python -m src.preprocess.run
	uv run python -m src.train.run
	@echo "[OK] pipeline concluído."

train-live:
	@echo "Rodando estagio de TREINO direto (sem DVC) com logs live..."
	uv run python -m src.train.run
	@echo "[OK] treino concluído."

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-train:
	@echo "Docker: Rodando pipeline no container (perfil train)..."
	docker compose -f docker/docker-compose.yml --env-file .env --profile train up --build train
	@echo "[OK] Pipeline concluído."
