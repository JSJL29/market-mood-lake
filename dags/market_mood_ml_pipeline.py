from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 7, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="market_mood_ml_pipeline",
    default_args=DEFAULT_ARGS,
    description="Export curated -> entraînement GRU -> enregistrement des métriques dans MongoDB",
    schedule=None,  # déclenchement manuel: le training est coûteux, pas à chaque ingestion API
    catchup=False,
    max_active_runs=1,
    tags=["market-mood-lake", "ml"],
) as dag:
    export_dataset = BashOperator(
        task_id="export_dataset",
        bash_command=(
            "mkdir -p /opt/airflow/data/curated_export /opt/airflow/models/model_runs && "
            "python -m src.ml.export_dataset "
            "--mongo_uri mongodb://mongodb:27017/ "
            "--collection market_sequences "
            "--output_path /opt/airflow/data/curated_export/market_sequences.npz"
        ),
    )

    train_price_only = BashOperator(
        task_id="train_price_only",
        bash_command=(
            "python -m src.ml.train_recorded "
            "--npz_path /opt/airflow/data/curated_export/market_sequences.npz "
            "--mode price_only "
            "--epochs 20 --batch_size 32 --lr 0.001 "
            "--mongo_uri mongodb://mongodb:27017/ "
            "--model_dir /opt/airflow/models/model_runs "
            "--run_note 'airflow manual ML DAG: price_only'"
        ),
    )

    train_full = BashOperator(
        task_id="train_full",
        bash_command=(
            "python -m src.ml.train_recorded "
            "--npz_path /opt/airflow/data/curated_export/market_sequences.npz "
            "--mode full "
            "--epochs 20 --batch_size 32 --lr 0.001 "
            "--mongo_uri mongodb://mongodb:27017/ "
            "--model_dir /opt/airflow/models/model_runs "
            "--run_note 'airflow manual ML DAG: full'"
        ),
    )

    export_dataset >> [train_price_only, train_full]
