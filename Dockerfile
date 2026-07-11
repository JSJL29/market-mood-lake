FROM apache/airflow:2.7.1

USER root

# Installation des dépendances système (build-essential requis pour Numba/LLVM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Création des dossiers nécessaires
RUN mkdir -p /opt/airflow/build /opt/airflow/data/raw /opt/airflow/scripts
RUN chown -R airflow:root /opt/airflow/build /opt/airflow/data/raw /opt/airflow/scripts

USER airflow

# Copie des requirements et installation
COPY build/requirements-airflow.txt /opt/airflow/build/requirements-airflow.txt
RUN pip install --no-cache-dir -r /opt/airflow/build/requirements-airflow.txt

# Copie une arborescence autonome identique aux chemins utilisés par les DAGs.
COPY build/*.py /opt/airflow/build/
COPY src/ /opt/airflow/src/
COPY src/ingestion/ /opt/airflow/scripts/ingestion/
COPY src/transform/ /opt/airflow/scripts/transform/
COPY scripts/ /opt/airflow/scripts_tools/
COPY dags/ /opt/airflow/dags/
