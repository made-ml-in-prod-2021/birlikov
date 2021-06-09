from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "shokhan",
    "email": ["birlikoff@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "2_dag_train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(5),
) as dag:

    preprocess = DockerOperator(
        image="shokhan/airflow-preprocess-data",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess-data",
        do_xcom_push=False,
        volumes=["/Users/shokhan/Desktop/MADE/Spring_2021/ml_prod/birlikov/airflow_ml_dags/data:/data"]
    )

    split = DockerOperator(
        image="shokhan/airflow-split-data",
        command="--data-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        volumes=["/Users/shokhan/Desktop/MADE/Spring_2021/ml_prod/birlikov/airflow_ml_dags/data:/data"]
    )

    train = DockerOperator(
        image="shokhan/airflow-train-model",
        command="--data-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        volumes=["/Users/shokhan/Desktop/MADE/Spring_2021/ml_prod/birlikov/airflow_ml_dags/data:/data"]
    )

    validate = DockerOperator(
        image="shokhan/airflow-validate-model",
        command="--data-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }} --metrics-dir /data/metrics/{{ ds }}",
        task_id="docker-airflow-validate-model",
        do_xcom_push=False,
        volumes=["/Users/shokhan/Desktop/MADE/Spring_2021/ml_prod/birlikov/airflow_ml_dags/data:/data"]
    )

    preprocess >> split >> train >> validate
