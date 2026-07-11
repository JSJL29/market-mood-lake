from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 7, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "market_mood_pipeline",
    default_args=default_args,
    description="Pipeline ETL pour le S&P500 + VIX + Fear&Greed",
    schedule=timedelta(hours=6),
    catchup=False,
    max_active_runs=1,
)

aws_env = {
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_DEFAULT_REGION": "us-east-1",
}

RAW_BUCKET = os.getenv("RAW_BUCKET", "raw")
MARKET_START_DATE = os.getenv("MARKET_START_DATE", "2015-01-01")
LABEL_HORIZON = int(os.getenv("LABEL_HORIZON", "1"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "30"))
XS_MAX_TICKERS = int(os.getenv("XS_MAX_TICKERS", "50"))
XS_MAX_SEQUENCES_PER_TICKER = int(os.getenv("XS_MAX_SEQUENCES_PER_TICKER", "1000"))

# Tâche 0 : acquiert le CSV (téléchargement configurable ou fichier local).
acquire_historical_source = BashOperator(
    task_id="acquire_historical_source",
    bash_command=(
        "python /opt/airflow/scripts_tools/acquire_historical_data.py "
        "--destination /opt/airflow/data/kaggle_stocks/SP500_Historical_Data.csv"
    ),
    env=aws_env,
    append_env=True,
    dag=dag,
)

# Tâche 1 : prépare la zone raw automatiquement
# - crée le bucket raw si absent
# - génère / upload sp500_combined.csv depuis data/kaggle_stocks/SP500_Historical_Data.csv
init_raw_sp500 = BashOperator(
    task_id="init_raw_sp500",
    bash_command=r"""
set -e

echo "Création du bucket raw si nécessaire..."

python - <<'PY'
import boto3

endpoint_url = "http://localstack:4566"
bucket = "raw"

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint_url,
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]

if bucket not in existing:
    s3.create_bucket(Bucket=bucket)
    print(f"Bucket créé: {bucket}")
else:
    print(f"Bucket déjà présent: {bucket}")
PY

echo "Vérification du fichier source S&P500..."

echo "Création / upload de sp500_combined.csv vers s3://raw..."

python /opt/airflow/build/unpack_to_raw.py \
  --input_dir /opt/airflow/data/kaggle_stocks \
  --input_file /opt/airflow/data/kaggle_stocks/SP500_Historical_Data.csv \
  --bucket_name raw \
  --output_file_name sp500_combined.csv \
  --endpoint-url http://localstack:4566

echo "Contenu de s3://raw après init:"
python - <<'PY'
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localstack:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

resp = s3.list_objects_v2(Bucket="raw")
for obj in resp.get("Contents", []):
    print(obj["Key"], obj["Size"])
PY
""",
    env=aws_env,
    dag=dag,
)

# Tâche 1 : ingestion de l'API VIX + Fear & Greed vers raw S3
fetch_market_mood = BashOperator(
    task_id="fetch_market_mood",
    bash_command=(
        "python /opt/airflow/scripts/ingestion/market_api.py "
        f"--bucket_name {RAW_BUCKET} "
        "--endpoint-url http://localstack:4566 "
        "--period max "
        f"--start_date {MARKET_START_DATE}"
    ),
    env=aws_env,
    dag=dag,
)

# Tâche 2 : transformation vers MySQL staging
preprocess_to_staging = BashOperator(
    task_id="preprocess_to_staging",
    bash_command=(
        "python /opt/airflow/scripts/transform/preprocess_to_staging.py "
        f"--bucket_raw {RAW_BUCKET} "
        "--sp500_file sp500_combined.csv "
        "--db_host mysql "
        "--db_user root "
        "--db_password root "
        "--endpoint-url http://localstack:4566 "
        f"--horizon {LABEL_HORIZON}"
    ),
    env=aws_env,
    dag=dag,
)

# Tâche 3 : chargement vers MongoDB curated
process_to_curated = BashOperator(
    task_id="process_to_curated",
    bash_command=(
        "python /opt/airflow/scripts/transform/process_to_curated.py "
        "--mysql_host mysql "
        "--mysql_user root "
        "--mysql_password root "
        "--mongo_uri mongodb://mongodb:27017/ "
        f"--window_size {WINDOW_SIZE}"
    ),
    dag=dag,
)

# La même exécution Airflow publie également la voie cross-sectional utilisée
# par DVC. Les limites évitent qu'une exécution d'évaluation sature la machine.
preprocess_to_staging_xs = BashOperator(
    task_id="preprocess_to_staging_xs",
    bash_command=(
        "cd /opt/airflow && python -m src.transform.preprocess_to_staging_xs "
        f"--bucket_raw {RAW_BUCKET} "
        "--stocks_file sp500_combined.csv "
        "--db_host mysql "
        "--db_user root "
        "--db_password root "
        "--endpoint-url http://localstack:4566 "
        f"--horizon {LABEL_HORIZON} "
        f"--max_tickers {XS_MAX_TICKERS}"
    ),
    env=aws_env,
    dag=dag,
)

process_to_curated_xs = BashOperator(
    task_id="process_to_curated_xs",
    bash_command=(
        "cd /opt/airflow && python -m src.transform.process_to_curated_xs "
        "--mysql_host mysql "
        "--mysql_user root "
        "--mysql_password root "
        "--mongo_uri mongodb://mongodb:27017/ "
        f"--window_size {WINDOW_SIZE} "
        f"--max_sequences_per_ticker {XS_MAX_SEQUENCES_PER_TICKER}"
    ),
    dag=dag,
)

acquire_historical_source >> init_raw_sp500 >> fetch_market_mood
fetch_market_mood >> preprocess_to_staging >> process_to_curated
fetch_market_mood >> preprocess_to_staging_xs >> process_to_curated_xs
