# Market Mood Lake

Data lake local consacré à l'analyse directionnelle du marché à partir de données historiques multi-actions, du VIX et du Fear & Greed Index.

Le projet met en œuvre :

- une zone **Raw** dans S3 via LocalStack ;
- une zone **Staging** dans MySQL ;
- une zone **Curated** dans MongoDB ;
- une orchestration complète avec Apache Airflow ;
- une API Gateway FastAPI ;
- un pipeline Machine Learning fondé sur des GRU ;
- une reproduction locale des transformations avec DVC.

---

## 1. Objectif

L'objectif est de déterminer si l'historique des prix et les indicateurs de sentiment permettent de prédire la direction future du marché : hausse ou baisse.

Le projet ne cherche pas uniquement à prévoir un prix exact. Il formule le problème comme une classification directionnelle, plus pertinente pour comparer le modèle à une baseline naïve.

Deux modes de données sont disponibles :

| Mode | Description |
|---|---|
| Mono-indice | Historique agrégé du S&P 500 |
| Cross-sectional | Historique de plusieurs actions du S&P 500 |

---

## 2. Périmètre d'automatisation

Le pipeline Airflow acquiert explicitement le dataset historique. Il réutilise le fichier local,
le télécharge depuis `HISTORICAL_DATA_URL`, ou génère en dernier recours un petit dataset de
démonstration déterministe. Une machine neuve peut donc exécuter le projet sans compte Kaggle.
Le fallback sert à valider l'infrastructure ; les résultats ML publiables doivent utiliser la
source historique complète.

| Étape | Statut | Responsable |
|---|---:|---|
| Acquisition du dataset historique | Automatique : fichier, URL, puis fallback de démonstration | Airflow / DVC |
| Vérification du CSV source | Automatique | Airflow / DVC |
| Création du bucket `raw` | Automatique | Airflow |
| Génération de `sp500_combined.csv` | Automatique | Airflow / DVC |
| Upload du fichier historique dans S3 | Automatique | Airflow / DVC |
| Collecte du VIX et du Fear & Greed Index | Automatique | Airflow / DVC |
| Transformation Raw vers Staging | Automatique | Airflow / DVC |
| Transformation Staging vers Curated | Automatique | Airflow / DVC |
| Entraînement GRU | Déclenchement manuel | DAG ML Airflow |
| Benchmark API | Manuel | Script de benchmark |

Pour une exécution hors ligne, le fichier attendu est :

```text
data/kaggle_stocks/SP500_Historical_Data.csv
```

Le fichier CSV n'est pas versionné dans Git mais devient une sortie DVC. Pour imposer une vraie
source sur une machine neuve, définir avant le démarrage :

```bash
export HISTORICAL_DATA_URL="https://.../SP500_Historical_Data.csv"
```

Sans cette variable, le fallback hors ligne est créé automatiquement. L'option
`--no-demo-fallback` permet aux exécutions de production d'échouer si la vraie source manque.

Avant toute exécution, le correcteur peut diagnostiquer son environnement avec une seule commande :

```bash
python scripts/doctor.py
```

Prévoir environ 10 Gio libres pour un premier build Docker (Airflow et PyTorch sont volumineux).
Une installation déjà construite est acceptée par le diagnostic à partir de 0,5 Gio, mais libérer
davantage d'espace reste recommandé.

---

## 3. Architecture

```text
Dataset historique                  API VIX / Fear & Greed
        |                                      |
        +------------------+-------------------+
                           |
                           v
                 Raw - S3 LocalStack
                           |
                           v
                   Staging - MySQL
                           |
                           v
                  Curated - MongoDB
                           |
                 +---------+---------+
                 |                   |
                 v                   v
             FastAPI             Pipeline GRU
```

| Zone | Technologie | Contenu |
|---|---|---|
| Raw | S3 LocalStack | CSV historique combiné et payloads JSON de sentiment |
| Staging | MySQL | Données nettoyées, enrichies et dédupliquées |
| Curated | MongoDB | Fenêtres temporelles, labels et résultats ML |

---

## 4. Services Docker

| Service | Rôle | Port |
|---|---|---:|
| `api` | API FastAPI | `8000` |
| `airflow-webserver` | Interface Airflow | `8081` |
| `airflow-scheduler` | Planification des DAGs | Interne |
| `localstack` | S3 local | `4566` |
| `mysql` | Zone Staging | `3306` |
| `mongodb` | Zone Curated | `27017` |
| `postgres` | Métadonnées Airflow | Interne |

---

## 5. Chemins de référence

Un seul chemin source est utilisé dans le dépôt :

```text
data/kaggle_stocks/SP500_Historical_Data.csv
```

Le chemin historique `data/kaggle_sp500` n'est plus utilisé.

### Exécution locale avec DVC

```text
src/ingestion/market_api.py
src/transform/preprocess_to_staging.py
src/transform/process_to_curated.py
data/kaggle_stocks/
http://localhost:4566
```

### Exécution dans Airflow

Docker monte les dossiers du dépôt dans le conteneur :

| Hôte | Conteneur Airflow |
|---|---|
| `./src` | `/opt/airflow/src` |
| `./build` | `/opt/airflow/build` |
| `./data` | `/opt/airflow/data` |
| `./models` | `/opt/airflow/models` |
| `./dags` | `/opt/airflow/dags` |

Les commandes du DAG doivent donc utiliser :

```text
/opt/airflow/src/ingestion/market_api.py
/opt/airflow/src/transform/preprocess_to_staging.py
/opt/airflow/src/transform/process_to_curated.py
```

Les noms réseau Docker sont :

```text
LocalStack : http://localstack:4566
MySQL     : mysql
MongoDB   : mongodb
```

---

## 6. Prérequis

- Docker Desktop ;
- WSL2 sous Windows ;
- Docker Compose ;
- Python 3.10 ou supérieur pour DVC et les scripts locaux ;
- le CSV source dans `data/kaggle_stocks/`.

Sous WSL, utiliser Python 3.11 et un environnement Linux dédié :

```bash
python3.11 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install -r build/requirements.txt
```

Credentials LocalStack :

```bash
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
```

---

## 7. Démarrage rapide

À la racine du projet :

```bash
docker compose up -d --build
docker compose ps
```

Interfaces :

```text
FastAPI : http://localhost:8000/docs
Airflow : http://localhost:8081
```

Identifiants Airflow :

```text
airflow / airflow
```

---

## 8. DAG principal autonome

Le DAG principal est :

```text
market_mood_pipeline
```

Il est planifié toutes les six heures et exécute :

```text
acquire_historical_source
    -> init_raw_sp500
    -> fetch_market_mood
    -> preprocess_to_staging
    -> process_to_curated
```

### `acquire_historical_source`

Cette tâche idempotente réutilise le CSV local ou le télécharge atomiquement depuis
`HISTORICAL_DATA_URL`, puis vérifie qu'il n'est pas vide.

### `init_raw_sp500`

Cette tâche :

1. crée le bucket `raw` s'il n'existe pas ;
2. vérifie la présence du fichier source ;
3. exécute `build/unpack_to_raw.py` ;
4. génère `sp500_combined.csv` ;
5. envoie le fichier dans `s3://raw/sp500_combined.csv`.

### `fetch_market_mood`

Cette tâche récupère :

- le VIX ;
- le Fear & Greed Index ;

puis écrit un fichier horodaté :

```text
s3://raw/market_mood_YYYYMMDDTHHMMSS.json
```

### `preprocess_to_staging`

Cette tâche :

- lit le CSV historique et les fichiers de sentiment dans S3 ;
- nettoie les données ;
- calcule les features ;
- construit le label directionnel ;
- charge les lignes dans MySQL.

### `process_to_curated`

Cette tâche :

- lit les données Staging ;
- construit des fenêtres temporelles de 30 jours ;
- prépare les documents ML ;
- écrit les séquences dans MongoDB.

---

## 9. Déclenchement et preuve d'exécution

Déclencher le DAG depuis l'interface Airflow ou avec :

```bash
docker compose exec airflow-scheduler   airflow dags trigger market_mood_pipeline
```

Afficher les derniers runs :

```bash
docker compose exec airflow-scheduler   airflow dags list-runs -d market_mood_pipeline --limit 5
```

Le run est valide lorsque les quatre tâches sont en état `success`.

### Vérifier la zone Raw

```bash
docker compose exec localstack   awslocal s3 ls s3://raw --recursive
```

Résultat attendu :

```text
sp500_combined.csv
market_mood_YYYYMMDDTHHMMSS.json
```

### Vérifier la zone Staging

```bash
docker compose exec mysql   mysql -uroot -proot -e   "SELECT COUNT(*) AS row_count FROM staging.market_data;"
```

### Vérifier la zone Curated

```bash
docker compose exec mongodb mongosh --quiet --eval '
const dbx = db.getSiblingDB("curated");
printjson({
  market_sequences: dbx.market_sequences.countDocuments({}),
  model_runs: dbx.model_runs.countDocuments({})
});
'
```

### Vérifier l'API

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/stats | python3 -m json.tool
```

Le healthcheck doit indiquer que les connexions S3, MySQL et MongoDB sont disponibles.

---

## 10. Reproduction avec DVC

DVC permet de relancer le pipeline depuis la machine hôte :

```bash
dvc repro
```

Ordre des étapes :

```text
acquire_historical_source
    -> unpack_to_raw
    -> fetch_market_mood
    -> preprocess_to_staging_xs
    -> process_to_curated_xs
```

La voie DVC est cross-sectional et volontairement bornée par défaut à 50 tickers et
1 000 séquences par ticker (`xs_max_tickers`, `xs_max_sequences_per_ticker` dans
`params.yaml`). Le DAG Airflow principal construit, lui, un proxy de marché
équipondéré mono-série.

Les bases restent les supports d'exécution, mais leur contenu utile est maintenant exporté sous
forme d'artefacts déterministes réellement suivis par DVC :

```text
data/kaggle_stocks/SP500_Historical_Data.csv
data/versioned_snapshots/market_data_xs.csv.gz
data/versioned_snapshots/market_sequences_xs.jsonl.gz
```

Les snapshots gzip ont un en-tête stable (`mtime=0`), un tri stable et un contenu canonique. DVC
peut ainsi mettre en cache, restaurer et comparer les données MySQL/MongoDB, et pas seulement un
fichier d'état. `always_changed` ne subsiste que sur les étapes qui interrogent ou publient un
système externe mutable (S3/API), pas sur le versionnement des bases.

Les paramètres locaux sont centralisés dans :

```text
params.yaml
```

DVC utilise les endpoints de la machine hôte :

```text
LocalStack : http://localhost:4566
MySQL     : localhost
MongoDB   : localhost
```

---

## 11. DAG Machine Learning

Le DAG ML est :

```text
market_mood_ml_pipeline
```

Il est déclenché manuellement, car l'entraînement est plus coûteux que l'ingestion.

```text
export_dataset
    -> train_price_only
    -> train_full
```

Sorties :

| Sortie | Emplacement |
|---|---|
| Dataset `.npz` | `data/curated_export/market_sequences.npz` |
| Artefacts JSON et checkpoints `.pt` | `models/model_runs/` |
| Résultats ML | `curated.model_runs` dans MongoDB |
| Dernier run | Endpoint `/stats` |

Les deux variantes sont :

| Mode | Features |
|---|---|
| `price_only` | Rendements, volatilité, RSI et moyennes mobiles |
| `full` | Features prix, VIX et Fear & Greed |

La validation est temporelle, regroupe les observations par date et purge les fenêtres
qui se chevauchent aux frontières. La classe de la baseline est apprise uniquement sur
le train. Chaque run enregistre le hash du dataset, la seed, le normaliseur, le schéma
des features et un checkpoint PyTorch réutilisable.

---

## 12. API Gateway

Documentation :

```text
http://localhost:8000/docs
```

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/health` | GET | État de l'API et des stockages |
| `/stats` | GET | Statistiques globales et derniers résultats |
| `/raw/` | GET | Objets de la zone Raw |
| `/staging/` | GET | Lignes de la zone Staging |
| `/curated/` | GET | Documents MongoDB |
| `/ingest` | POST | Ingestion de référence |
| `/ingest_fast` | POST | Ingestion batch optimisée |

Pour afficher les runs ML :

```bash
curl -s   "http://localhost:8000/curated/?collection=model_runs&limit=5"   | python3 -m json.tool
```

---

## 13. Benchmark d'ingestion

Lancer :

```bash
python3 scripts/benchmark_ingest.py   --runs 10   --batch-sizes 1 100
```

Le mode rapide possède un chemin dédié aux micro-batches : il évite la compilation JIT inutile
pour une ligne et réutilise un pool de connexions MySQL. Le seuil de 30 % reste mesuré par le
benchmark (moyenne et médiane), car un objectif de latence dépend aussi de la machine et des
services Docker. Les calculs et ce chemin batch=1 sont couverts par des tests de non-régression.

Le benchmark utilise `127.0.0.1` afin d'éviter le délai de fallback IPv6 de `localhost`
et écrit dans une table dédiée. Sur la validation du 11 juillet 2026, les gains moyens
observés étaient de 43 % pour un batch de 1 et 59 % pour un batch de 100.

Résultats :

```text
data/benchmarks/ingest_benchmark_results.json
```

---

## 14. Structure du projet

```text
market-mood-lake/
├── build/
│   └── unpack_to_raw.py
├── dags/
│   ├── market_mood_pipeline.py
│   └── market_mood_ml_pipeline.py
├── data/
│   ├── kaggle_stocks/
│   │   └── SP500_Historical_Data.csv
│   ├── curated_export/
│   ├── benchmarks/
│   └── .pipeline_state/
├── models/
│   └── model_runs/
├── scripts/
│   ├── benchmark_ingest.py
│   └── smoke_api.py
├── src/
│   ├── ingestion/
│   ├── transform/
│   ├── api/
│   └── ml/
├── tests/
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
└── README.md
```

---

## 15. Checklist de démonstration

```text
[ ] Le CSV source existe dans data/kaggle_stocks/
[ ] Les services Docker sont Up ou healthy
[ ] Le DAG market_mood_pipeline est vert
[ ] s3://raw contient sp500_combined.csv
[ ] s3://raw contient au moins un market_mood_*.json
[ ] staging.market_data contient des lignes
[ ] curated.market_sequences contient des documents
[ ] /health indique S3, MySQL et MongoDB disponibles
[ ] /stats retourne les volumes du data lake
[ ] Le DAG ML produit des artefacts dans models/model_runs/
```

---

## 16. Limites connues

- L'acquisition initiale nécessite soit un fichier local, soit une URL autorisée dans `HISTORICAL_DATA_URL`.
- L'entraînement ML est volontairement séparé du DAG d'ingestion.
- DVC orchestre des écritures vers des services externes ; ses fichiers d'état ne remplacent pas le versionnement natif d'une base de données.
- Sans `sector_etfs.csv` et `ticker_sector_map.csv`, les colonnes sectorielles XS sont
  explicitement neutres et le mode ML `sector_full` est refusé.
- Le disque hébergeant Docker doit conserver plusieurs gigaoctets libres ; les images
  scientifiques et les volumes MongoDB cross-sectional sont volumineux.
- Le signal directionnel observé par les modèles reste faible et instable selon les périodes.

---

## 17. Conclusion

Market Mood Lake fournit une chaîne démontrable de bout en bout :

```text
CSV historique + API
        -> Raw S3
        -> Staging MySQL
        -> Curated MongoDB
        -> API et modèles GRU
```

Après le dépôt initial du CSV source, Airflow prend en charge automatiquement l'initialisation du Raw, l'ingestion des données de sentiment et les transformations vers Staging puis Curated. DVC fournit un second chemin d'exécution local cohérent avec les mêmes scripts et les mêmes paramètres.
