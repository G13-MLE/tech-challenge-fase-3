"""Testes da DAG Airflow de retreino."""

import pytest

airflow = pytest.importorskip("airflow")


def test_dag_imports():
    """DAG é importável e contém as tasks esperadas."""
    from dags.train_pipeline import dag

    assert dag.dag_id == "train_pipeline"
    task_ids = {t.task_id for t in dag.tasks}
    assert "load_data_task" in task_ids
    assert "train_model_task" in task_ids
    assert "save_model_task" in task_ids
    assert "evaluate_model_task" in task_ids


def test_dag_catchup_false():
    """DAG tem catchup=False."""
    from dags.train_pipeline import dag

    assert dag.catchup is False


def test_dag_default_args():
    """DAG tem retries e retry_exponential_backoff configurados."""
    from dags.train_pipeline import dag

    defaults = dag.default_args
    assert defaults["retries"] >= 2
    assert defaults["retry_exponential_backoff"] is True
