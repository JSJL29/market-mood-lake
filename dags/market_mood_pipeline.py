from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'market_mood_pipeline',
    default_args=default_args,
    description='Pipeline ETL pour le S&P500 + VIX + Fear&Greed',
    schedule=timedelta(hours=6),
    catchup=False,  # ne pas rattraper les créneaux passés depuis start_date
    max_active_runs=1,  # une seule exécution à la fois
)

# Tâche 1 : ingestion de l'API VIX + Fear & Greed vers raw (S3)
# Le CSV Kaggle S&P500, lui, est ingéré une seule fois manuellement via build/unpack_to_raw.py
fetch_market_mood = BashOperator(
    task_id='fetch_market_mood',
    bash_command='python /opt/airflow/scripts/ingestion/market_api.py --endpoint-url http://localstack:4566 --period max --start_date 2015-01-01',
    dag=dag,
)

# Tâche 2 : transformation vers MySQL (staging)
preprocess_to_staging = BashOperator(
    task_id='preprocess_to_staging',
    bash_command=(
        'python /opt/airflow/scripts/transform/preprocess_to_staging.py '
        '--bucket_raw raw '
        '--sp500_file sp500_combined.csv '
        '--db_host mysql --db_user root --db_password root '
        '--endpoint-url http://localstack:4566'
    ),
    dag=dag,
)

# Tâche 3 : chargement vers MongoDB (curated)
process_to_curated = BashOperator(
    task_id='process_to_curated',
    bash_command=(
        'python /opt/airflow/scripts/transform/process_to_curated.py '
        '--mysql_host mysql --mysql_user root --mysql_password root '
        '--mongo_uri mongodb://mongodb:27017/ '
        '--window_size 30'
    ),
    dag=dag,
)

fetch_market_mood >> preprocess_to_staging >> process_to_curated