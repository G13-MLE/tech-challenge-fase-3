FROM apache/airflow:3.3.1-python3.14

USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /tmp/deps/
RUN cd /tmp/deps && uv export --frozen --no-dev --no-emit-project -o /tmp/deps/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/deps/requirements.txt \
    && rm -rf /tmp/deps
USER airflow

COPY src/ /opt/airflow/src/
COPY configs/ /opt/airflow/configs/
ENV PYTHONPATH=/opt/airflow
