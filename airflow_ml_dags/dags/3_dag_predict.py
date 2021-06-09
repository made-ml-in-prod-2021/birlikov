from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

default_args = {
    "owner": "shokhan",
    "email": ["birlikoff@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "3_dag_predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(3),
) as dag:

    model_path = Variable.get("model_path")

    get_predictions = DockerOperator(
        image="shokhan/airflow-get-predictions",
        command="--data-dir /data/raw/{{ ds }} --predictions-dir /data/predictions/{{ ds }}",
        task_id="docker-airflow-get-predictions",
        do_xcom_push=False,
        environment={
            'MODEL_PATH': model_path
        },
        volumes=["/Users/shokhan/Desktop/MADE/Spring_2021/ml_prod/birlikov/airflow_ml_dags/data:/data"]
    )

    notify = BashOperator(
        task_id="notify",
        bash_command='echo "Got new predictions for today!"',
    )

    get_predictions >> notify