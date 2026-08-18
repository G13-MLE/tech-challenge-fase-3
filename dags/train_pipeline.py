"""DAG Airflow de retreino do modelo de triagem.

Orquestra as etapas de carregamento, treinamento e validação do modelo,
reutilizando as funções do pipeline DVC via src.orchestration.training_tasks.
XCom passa apenas PATHs (strings), nunca objetos.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.sdk import task

from src.orchestration.training_tasks import load_data, save_model, train_model

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
    def load_data_task() -> str:
        """Lê o CSV processado e retorna o caminho."""
        return load_data()

    @task(retries=3, retry_exponential_backoff=True, execution_timeout=timedelta(hours=1))
    def train_model_task(csv_path: str) -> str:
        """Treina o pipeline e salva modelo + hash."""
        return train_model(csv_path)

    @task(retries=2, retry_exponential_backoff=True, execution_timeout=timedelta(minutes=30))
    def save_model_task(model_path: str) -> str:
        """Valida integridade do modelo salvo (hash + classes)."""
        return save_model(model_path)

    csv_path = load_data_task()
    model_path = train_model_task(csv_path)
    save_model_task(model_path)


if __name__ == "__main__":
    dag.test()
