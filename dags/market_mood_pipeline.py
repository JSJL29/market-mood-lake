from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

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

# Tâche 0 : prépare la zone raw automatiquement
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

if [ ! -f "/opt/airflow/data/kaggle_stocks/SP500_Historical_Data.csv" ]; then
  echo "ERREUR: fichier manquant: /opt/airflow/data/kaggle_stocks/SP500_Historical_Data.csv"
  echo "Place ton CSV Kaggle dans: data/kaggle_stocks/SP500_Historical_Data.csv"
  exit 1
fi

echo "Création / upload de sp500_combined.csv vers s3://raw..."

python /opt/airflow/build/unpack_to_raw.py \
  --input_dir /opt/airflow/data/kaggle_stocks \
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
        "--endpoint-url http://localstack:4566 "
        "--period max "
        "--start_date 2015-01-01"
    ),
    env=aws_env,
    dag=dag,
)

# Tâche 2 : transformation vers MySQL staging
preprocess_to_staging = BashOperator(
    task_id="preprocess_to_staging",
    bash_command=(
        "python /opt/airflow/scripts/transform/preprocess_to_staging.py "
        "--bucket_raw raw "
        "--sp500_file sp500_combined.csv "
        "--db_host mysql "
        "--db_user root "
        "--db_password root "
        "--endpoint-url http://localstack:4566"
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
        "--window_size 30"
    ),
    dag=dag,
)

init_raw_sp500 >> fetch_market_mood >> preprocess_to_staging >> process_to_curated