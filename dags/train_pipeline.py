"""DAG Airflow de retreino do modelo de triagem.

Orquestra as etapas de ingestão, treinamento, validação, avaliação e
exportação ONNX, reutilizando as funções do pipeline DVC via
src.orchestration.training_tasks. XCom passa apenas PATHs (strings), nunca objetos.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.sdk import task

from src.orchestration.training_tasks import (
    evaluate_model,
    export_onnx_model,
    ingest_data,
    save_model,
    train_model,
)

default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    description="Retreino semanal do modelo de triagem (TF-IDF + RandomForest)",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["triagem", "treino"],
) as dag:

    @task(retries=3, retry_exponential_backoff=True, execution_timeout=timedelta(hours=1))
    def ingest_data_task() -> str:
        """Executa ingestão: validação + pré-processamento dos laudos brutos."""
        return ingest_data()

    @task(retries=3, retry_exponential_backoff=True, execution_timeout=timedelta(hours=1))
    def train_model_task(csv_path: str) -> str:
        """Treina o pipeline e salva modelo + hash."""
        return train_model(csv_path)

    @task(retries=2, retry_exponential_backoff=True, execution_timeout=timedelta(minutes=30))
    def save_model_task(model_path: str) -> str:
        """Valida integridade do modelo salvo (hash + classes)."""
        return save_model(model_path)

    @task(retries=2, retry_exponential_backoff=True, execution_timeout=timedelta(minutes=30))
    def evaluate_model_task(model_path: str) -> str:
        """Executa a avaliação do modelo sobre o split de teste."""
        return evaluate_model(model_path)

    @task(retries=2, retry_exponential_backoff=True, execution_timeout=timedelta(minutes=30))
    def export_onnx_task(model_path: str) -> str:
        """Exporta o modelo para ONNX e verifica paridade de predição."""
        return export_onnx_model(model_path)

    processed_path = ingest_data_task()
    model_path = train_model_task(processed_path)
    validated_model_path = save_model_task(model_path)
    evaluate_model_task(validated_model_path)
    export_onnx_task(validated_model_path)


if __name__ == "__main__":
    dag.test()
