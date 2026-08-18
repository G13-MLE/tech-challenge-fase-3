# =============================================================================
# Estágio 1: Build - instala dependências com uv
# =============================================================================
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Estágio 2: Runtime - imagem leve com apenas o necessário
# =============================================================================
FROM python:3.14-slim AS runtime

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv
COPY src/ /app/src/
COPY models/model.joblib /app/models/model.joblib

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
