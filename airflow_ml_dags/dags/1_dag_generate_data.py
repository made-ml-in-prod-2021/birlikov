from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "shokhan",
    "email": ["birlikoff@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "1_dag_generate_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    generate = DockerOperator(
        image="shokhan/airflow-generate-data",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-generate-data",
        do_xcom_push=False,
        volumes=["/Users/shokhan/Desktop/MADE/Spring_2021/ml_prod/birlikov/airflow_ml_dags/data:/data"]
    )

    notify = BashOperator(
        task_id="notify",
        bash_command='echo "Generated new data!"',
    )

    generate >> notify