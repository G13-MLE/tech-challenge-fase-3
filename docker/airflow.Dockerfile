FROM apache/airflow:3.3.1-python3.14

USER root
RUN python -m pip install --no-cache-dir \
    "scikit-learn>=1.6.0" \
    "joblib>=1.4.0" \
    "pydantic-settings>=2.8.0" \
    "pyyaml>=6.0.0" \
    "python-dotenv>=1.0.0"
USER airflow

COPY src/ /opt/airflow/src/
COPY configs/ /opt/airflow/configs/
ENV PYTHONPATH=/opt/airflow
