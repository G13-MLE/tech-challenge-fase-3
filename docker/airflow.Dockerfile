FROM apache/airflow:3.3.1-python3.14

USER root
RUN pip install --no-cache-dir scikit-learn>=1.6.0 joblib>=1.4.0
USER airflow

COPY src/ /opt/airflow/src/
ENV PYTHONPATH=/opt/airflow
