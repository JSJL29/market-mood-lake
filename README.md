# Market Mood Lake

Pipeline local de données et de Machine Learning consacré à l’analyse directionnelle du marché à partir de prix historiques, du VIX et du Fear & Greed Index.

Le projet fournit :

- une zone Raw compatible S3 avec LocalStack ;
- une zone Staging dans MySQL ;
- une zone Curated dans MongoDB ;
- deux pipelines Airflow, ETL et ML ;
- un DAG DVC fondé sur de vrais artefacts fichiers ;
- une API FastAPI de consultation et d’ingestion ;
- des modèles GRU et des validations chronologiques ;
- une démonstration reproductible en une commande.

## Démarrage rapide sous WSL

Prérequis :

- WSL 2 avec Python 3.10 ou plus récent ;
- Docker Desktop avec l’intégration WSL activée ;
- environ 10 Gio libres pour le premier build.

Depuis la racine du dépôt :

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install -r build/requirements.txt
python scripts/doctor.py
docker compose up -d --build
```

Si la commande `docker compose` de WSL n’est pas disponible, Docker Desktop peut être appelé avec `docker.exe compose`.

Interfaces locales :

| Service | Adresse | Identifiants |
|---|---|---|
| API | http://localhost:8000 | aucun |
| Swagger | http://localhost:8000/docs | aucun |
| Airflow | http://localhost:8081 | `airflow` / `airflow` |
| LocalStack S3 | http://localhost:4566 | clés locales `test` / `test` |
| MySQL | `localhost:3306` | `root` / `root` |
| MongoDB | `localhost:27017` | aucun |

Vérifier les conteneurs :

```bash
docker compose ps
```

Tous les services applicables doivent être `healthy`.

## Démonstration reproductible

La démonstration destinée au correcteur s’exécute sous WSL/Linux avec une seule commande :

```bash
python scripts/run_reproducible_demo.py
```

Elle enchaîne :

1. le diagnostic de l’environnement ;
2. le démarrage des services Docker ;
3. la suite de tests Python ;
4. `dvc repro` ;
5. le smoke test de l’API et des volumes ;
6. le benchmark apparié des endpoints d’ingestion.

Le rapport final contient le commit Git, les commandes exécutées, leurs durées, leur statut et l’empreinte SHA-256 du benchmark :

```text
data/benchmarks/reproducible_demo_report.json
```

Pour conserver l’état DVC existant et relancer seulement les contrôles rapides :

```bash
python scripts/run_reproducible_demo.py --skip-dvc
```

## Architecture

```text
Source historique                  VIX / Fear & Greed
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              Raw — LocalStack S3
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Proxy mono-série     Données multi-tickers
             │                   │
             └─────────┬─────────┘
                       ▼
               Staging — MySQL
                       │
                       ▼
              Curated — MongoDB
                  ┌────┴────┐
                  ▼         ▼
               FastAPI   Pipeline ML
```

Deux représentations sont construites :

| Voie | Description | Tables/collections |
|---|---|---|
| Mono-série | Proxy de marché équipondéré | `market_data`, `market_sequences` |
| Cross-sectional | Fenêtres séparées par ticker | `market_data_xs`, `market_sequences_xs` |

Le DAG Airflow principal exécute les deux voies après une acquisition commune. La voie DVC est cross-sectional et bornée par défaut pour rester compatible avec une machine d’évaluation.

## Acquisition historique autonome

L’étape `acquire_historical_source` applique cet ordre :

1. réutiliser un CSV valide déjà présent ;
2. télécharger l’URL définie dans `HISTORICAL_DATA_URL` ;
3. en l’absence des deux, générer une fixture déterministe hors ligne de 3 tickers et 520 séances.

Le projet peut donc démarrer sur une machine neuve sans compte Kaggle, sans URL et sans copie manuelle de fichier.

LocalStack ne monte pas directement ce CSV depuis l’hôte : son hook d’initialisation crée uniquement
le bucket `raw`. Le fichier est acquis puis publié par Airflow ou DVC, ce qui garantit que le fallback
est réellement exercé sur un clone vierge.

Pour utiliser une vraie source historique :

```bash
export HISTORICAL_DATA_URL="https://example.org/SP500_Historical_Data.csv"
dvc repro -f acquire_historical_source
dvc repro
```

Le CSV doit contenir au minimum les colonnes `ticker`, `date` et `close`. La fixture intégrée valide l’infrastructure et les traitements ; elle ne doit pas être présentée comme un dataset financier réel pour interpréter les performances ML.

Pour interdire le fallback de démonstration dans un contexte de production :

```bash
python scripts/acquire_historical_data.py \
  --destination data/kaggle_stocks/SP500_Historical_Data.csv \
  --no-demo-fallback
```

## Pipeline DVC

Lancer la reproduction :

```bash
dvc repro
```

Graphe principal :

```text
acquire_historical_source
    → unpack_to_raw
    → fetch_market_mood
    → preprocess_to_staging_xs
    → process_to_curated_xs
```

Les paramètres sont centralisés dans [params.yaml](params.yaml), notamment :

```yaml
horizon: 1
window_size: 30
xs_max_tickers: 50
xs_max_sequences_per_ticker: 1000
```

### Artefacts réellement suivis

DVC ne se limite plus à des fichiers d’état. Chaque zone produit un artefact de contenu :

| Étape | Sortie DVC |
|---|---|
| Acquisition | `data/kaggle_stocks/SP500_Historical_Data.csv` |
| Raw historique | `data/raw_snapshot/sp500_combined.csv` |
| Raw sentiment | `data/raw_snapshot/market_mood.json` |
| Staging MySQL | `data/versioned_snapshots/market_data_xs.csv.gz` |
| Curated MongoDB | `data/versioned_snapshots/market_sequences_xs.jsonl.gz` |

Les snapshots gzip utilisent un en-tête stable, un tri stable et un JSON canonique. Le DAG ne contient plus de `always_changed: true` ni de fichier `.pipeline_state`.

Un second appel sans modification doit afficher :

```text
Stage 'acquire_historical_source' didn't change, skipping
...
Data and pipelines are up to date.
```

Les appels externes sont ainsi figés par leur snapshot. Pour rafraîchir volontairement le VIX et le Fear & Greed :

```bash
dvc repro -f fetch_market_mood
```

Le dépôt ne configure pas de stockage DVC distant par défaut. Sur un clone neuf, les étapes reconstruisent les artefacts localement ; une équipe peut ajouter son propre remote avec `dvc remote add` puis `dvc push`.

## Airflow

Deux DAGs sont disponibles.

### `market_mood_pipeline`

Pipeline ETL planifié toutes les six heures :

```text
acquire_historical_source
    → init_raw_sp500
    → fetch_market_mood
        ├→ preprocess_to_staging    → process_to_curated
        └→ preprocess_to_staging_xs → process_to_curated_xs
```

Déclenchement manuel :

```bash
docker compose exec airflow-scheduler \
  airflow dags trigger market_mood_pipeline
```

Contrôler les erreurs d’import :

```bash
docker compose exec airflow-scheduler \
  airflow dags list-import-errors
```

### `market_mood_ml_pipeline`

DAG manuel et séparé, car l’entraînement est plus coûteux :

```text
export_dataset
    ├→ train_price_only
    └→ train_full
```

Déclenchement :

```bash
docker compose exec airflow-scheduler \
  airflow dags trigger market_mood_ml_pipeline
```

Les métriques sont enregistrées dans `curated.model_runs` et les checkpoints dans `models/model_runs/`.

## Transformations et qualité des données

Les transformations calculent notamment :

- rendement logarithmique ;
- volatilité glissante sur 20 séances ;
- RSI sur 14 séances ;
- ratio de moyennes mobiles ;
- momentum ;
- VIX et Fear & Greed alignés uniquement vers le passé ;
- cible directionnelle à horizon configurable ;
- contexte sectoriel lorsqu’il est disponible.

Les publications MySQL et MongoDB utilisent des remplacements atomiques pour éviter les états partiels. Les fenêtres sont construites ticker par ticker, sans traverser les séries ni utiliser d’information future.

## API FastAPI

Routes principales :

| Méthode | Route | Fonction |
|---|---|---|
| GET | `/health` | état API et connexions |
| GET | `/stats` | volumes Raw, Staging et Curated |
| GET | `/raw/` | objets Raw |
| GET | `/staging/` | données MySQL |
| GET | `/curated/` | séquences MongoDB |
| POST | `/ingest` | mini-pipeline Raw → Staging → Curated, insertion ligne par ligne |
| POST | `/ingest_fast` | même mini-pipeline, calcul vectorisé et insertion batch |
| DELETE | `/ingest/benchmark-data` | nettoyage de la table de benchmark isolée |

Exemple :

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
```

Pour un appel normal, le payload est d’abord archivé dans `s3://raw/api_ingest/`, puis transformé
dans `staging.market_data_ingest`. À partir de 31 observations, les fenêtres de 30 jours et leur
label suivant sont publiés atomiquement dans `curated.market_sequences_ingest`. Ces objets sont
consultables avec :

```bash
curl "http://127.0.0.1:8000/staging/?table=market_data_ingest"
curl "http://127.0.0.1:8000/curated/?collection=market_sequences_ingest"
```

Les appels avec `"benchmark": true` restent volontairement isolés : ils écrivent uniquement dans
`market_data_ingest_benchmark`, ne créent pas d’objet Raw et ne reconstruisent pas Curated. Ils ne
modifient jamais `market_data` ni les collections produites par Airflow/DVC.

## Machine Learning

Le pipeline ML :

- exporte MongoDB vers un snapshot NPZ ;
- sépare les dates chronologiquement avec purge entre les ensembles ;
- apprend la normalisation uniquement sur le train ;
- compare le modèle à une baseline calculée sur le train ;
- fixe les graines aléatoires ;
- enregistre le hash du dataset, le schéma de features, les métriques et le checkpoint PyTorch.

Modes principaux :

- `price_only` : prix et indicateurs techniques ;
- `full` : prix, indicateurs et sentiment ;
- `sector_full` : contexte sectoriel, uniquement si les sources sectorielles existent.

Les scores obtenus sur la fixture de démonstration servent à valider le code, pas à conclure sur la prédictibilité du marché.

## Tests et contrôles

Suite Python :

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Compilation sans écrire dans les dossiers montés Windows :

```bash
PYTHONPYCACHEPREFIX=/tmp/market-mood-pycache \
  python -m compileall -q src scripts tests dags build
```

Smoke test :

```bash
python scripts/smoke_api.py
```

Autres contrôles utiles :

```bash
python scripts/doctor.py
docker compose config --quiet
dvc status
git diff --check
```

## Benchmark d’ingestion

Commande :

```bash
python scripts/benchmark_ingest.py --runs 10 --batch-sizes 1 100
```

Le protocole :

- vide uniquement la table de benchmark ;
- chauffe les deux endpoints hors mesure ;
- utilise des lignes SQL fraîches ;
- alterne l’ordre normal/rapide et rapide/normal ;
- sépare le temps HTTP du temps serveur ;
- calcule les gains par paires ;
- exige une amélioration médiane d’au moins 30 % pour chaque batch ;
- retourne un code non nul si une exigence échoue.

Résultats détaillés :

```text
data/benchmarks/ingest_benchmark_results.json
```

Les durées dépendent de la machine ; le protocole et les mesures brutes sont conservés pour rendre le résultat vérifiable.

## Structure utile

```text
market-mood-lake/
├── build/                 # image Airflow et préparation Raw
├── dags/                  # DAGs ETL et ML
├── data/
│   ├── benchmarks/        # rapports de démonstration
│   ├── curated_export/    # snapshots NPZ ML
│   ├── kaggle_stocks/     # source acquise/générée
│   ├── raw_snapshot/      # sorties Raw DVC
│   └── versioned_snapshots/ # exports MySQL/MongoDB DVC
├── models/model_runs/     # checkpoints et métriques ML
├── scripts/               # diagnostic, démo, benchmark, snapshots
├── src/
│   ├── api/
│   ├── ingestion/
│   ├── ml/
│   └── transform/
├── tests/
├── docker-compose.yml
├── dvc.yaml
└── params.yaml
```

## Limites connues

- La fixture hors ligne est synthétique et volontairement petite.
- Le rafraîchissement du VIX et du Fear & Greed nécessite un accès réseau ; leur snapshot DVC permet ensuite de reproduire le run sans nouvel appel.
- Les données sectorielles sont optionnelles. Sans elles, les colonnes sectorielles sont neutres et `sector_full` est refusé.
- L’entraînement ML reste séparé du DAG ETL pour contrôler son coût.
- Les images Airflow, PyTorch et les bases nécessitent plusieurs gigaoctets de disque.
- Le signal directionnel observé peut être faible ou instable ; le projet ne constitue pas un conseil financier.

## Arrêt et nettoyage

Arrêter les services sans supprimer les données :

```bash
docker compose stop
```

Supprimer les conteneurs en conservant les volumes :

```bash
docker compose down
```

La suppression des volumes efface MySQL, MongoDB, PostgreSQL et LocalStack ; ne l’utiliser que si une réinitialisation totale est souhaitée :

```bash
docker compose down -v
```
