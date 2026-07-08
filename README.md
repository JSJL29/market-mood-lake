# Market Mood Lake

Data lake complet pour tester la prédiction directionnelle du S&P 500 à partir de données de marché et d'indicateurs de sentiment.

Le projet combine :

- un historique fichier du S&P 500 / actions S&P 500 ;
- une source API pour le VIX et le Fear & Greed Index ;
- une architecture `raw / staging / curated` ;
- une orchestration Apache Airflow ;
- une API Gateway FastAPI ;
- un benchmark avancé `/ingest` vs `/ingest_fast` ;
- un pipeline ML automatisé avec GRU et stockage des résultats dans MongoDB.

---

## 1. Résumé du projet

L'objectif métier est de vérifier si la direction future du S&P 500, hausse ou baisse, peut être prédite à partir de l'historique de prix combiné à deux indicateurs de sentiment de marché :

- le **VIX**, indicateur de volatilité implicite ;
- le **Fear & Greed Index**, indicateur synthétique de sentiment de marché.

Le projet ne cherche pas à prédire le prix exact du marché, car une prédiction naïve du type « prix de demain ≈ prix d'aujourd'hui » peut déjà donner un RMSE faible sans réelle utilité décisionnelle. Le problème est donc formulé comme une **classification directionnelle** : prédire si le marché monte ou baisse à un horizon donné.

Deux variantes existent :

| Variante | Description |
|---|---|
| Mono-indice | Pipeline historique sur le S&P 500 uniquement. |
| Cross-sectional | Pipeline multi-actions sur plusieurs tickers du S&P 500 pour augmenter le nombre d'échantillons d'entraînement. |

---

## 2. Architecture

Le data lake suit une architecture en trois zones.

| Zone | Technologie | Contenu |
|---|---|---|
| Raw | S3 LocalStack | CSV bruts, fichier `sp500_combined.csv`, payloads JSON VIX / Fear & Greed. |
| Staging | MySQL | Tables tabulaires nettoyées, enrichies et dédupliquées. |
| Curated | MongoDB + artefacts `.npz` | Séquences ML, runs d'entraînement, benchmarks et métriques finales. |

Flux principal :

```text
CSV Kaggle / API Market Mood
        |
        v
Raw S3 LocalStack
        |
        v
Staging MySQL
        |
        v
Curated MongoDB + exports .npz
        |
        v
FastAPI + Airflow + ML GRU
```

Services Docker :

| Service | Rôle | Port local |
|---|---|---|
| `api` | FastAPI Gateway | `8000` |
| `airflow-webserver` | Interface Airflow | `8081` |
| `airflow-scheduler` | Scheduler Airflow | interne |
| `localstack` | S3 local | `4566` |
| `mysql` | Staging SQL | `3306` |
| `mongodb` | Curated NoSQL | `27017` |
| `postgres` | Metadata Airflow | interne |

---

## 3. Prérequis

Sous Windows, utiliser WSL2 avec Docker Desktop.

Prérequis :

- Docker Desktop avec intégration WSL2 activée ;
- Ubuntu / WSL2 ;
- Python 3.10+ pour les scripts locaux ;
- `curl` ;
- dataset CSV placé dans le bon dossier.

Le fichier source attendu pour l'initialisation automatique est :

```text
data/kaggle_stocks/SP500_Historical_Data.csv
```

Le DAG principal se charge ensuite de créer automatiquement dans S3 :

```text
s3://raw/sp500_combined.csv
```

---

## 4. Credentials locaux

Les credentials AWS utilisés avec LocalStack sont factices. LocalStack les accepte, mais `boto3` exige qu'ils existent.

| Service | Identifiants |
|---|---|
| LocalStack S3 | `AWS_ACCESS_KEY_ID=test`, `AWS_SECRET_ACCESS_KEY=test`, `AWS_DEFAULT_REGION=us-east-1` |
| Airflow | `airflow` / `airflow` |
| MySQL | `root` / `root` |
| MongoDB | pas d'authentification locale |

Pour un terminal local :

```bash
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
```

---

## 5. Installation rapide

Depuis WSL, à la racine du projet :

```bash
docker compose up -d --build
```

Vérifier les conteneurs :

```bash
docker compose ps
```

Tous les services principaux doivent être `Up` ou `healthy`.

Interface API :

```text
http://localhost:8000/docs
```

Interface Airflow :

```text
http://localhost:8081
```

Identifiants Airflow :

```text
airflow / airflow
```

---

## 6. Lancement du pipeline complet

### 6.1 DAG principal : data lake

Dans Airflow, déclencher :

```text
market_mood_pipeline
```

Le DAG exécute maintenant le pipeline complet en autonomie :

```text
init_raw_sp500
    -> fetch_market_mood
    -> preprocess_to_staging
    -> process_to_curated
```

Détail des tâches :

| Tâche | Rôle |
|---|---|
| `init_raw_sp500` | Crée le bucket `raw` si absent, lit `data/kaggle_stocks/SP500_Historical_Data.csv`, génère et upload `sp500_combined.csv` dans S3. |
| `fetch_market_mood` | Récupère VIX + Fear & Greed et ajoute un JSON horodaté dans S3. |
| `preprocess_to_staging` | Lit les fichiers raw, calcule les features et alimente MySQL. |
| `process_to_curated` | Transforme les données staging en données curated. |

Après exécution, vérifier S3 :

```bash
docker compose exec localstack awslocal s3 ls s3://raw --recursive
```

Résultat attendu :

```text
sp500_combined.csv
market_mood_YYYYMMDDTHHMMSS.json
```

### 6.2 DAG ML

Dans Airflow, déclencher ensuite :

```text
market_mood_ml_pipeline
```

Le DAG ML exécute :

```text
export_dataset
    -> train_price_only
    -> train_full
```

Il produit :

- un export `.npz` dans `data/curated_export/` ;
- des artefacts JSON dans `models/model_runs/` ;
- des documents MongoDB dans `curated.model_runs` ;
- des métriques visibles dans `/stats`.

---

## 7. Vérifications locales

### 7.1 Healthcheck

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Résultat attendu :

```json
{
  "api_status": "healthy",
  "connections": {
    "s3": true,
    "mysql": true,
    "mongodb": true
  }
}
```

### 7.2 Statistiques globales

```bash
curl -s http://localhost:8000/stats | python3 -m json.tool
```

L'endpoint doit afficher :

- le nombre d'objets raw ;
- les volumes MySQL staging ;
- les collections MongoDB curated ;
- le dernier run ML ;
- les benchmarks si disponibles ;
- les erreurs d'ingestion ou de data quality si elles existent.

Exemple d'état attendu :

```text
raw.object_count > 0
staging.market_data.row_count > 0
curated.model_runs.document_count > 0
ml.latest_model_run != null
```

### 7.3 Endpoints principaux

```bash
curl -s http://localhost:8000/raw/ | python3 -m json.tool
curl -s http://localhost:8000/staging/ | python3 -m json.tool
curl -s "http://localhost:8000/curated/?collection=model_runs&limit=5" | python3 -m json.tool
```

Remarque : l'endpoint `/curated/` lit par défaut `market_sequences`. Pour afficher les résultats ML, utiliser explicitement :

```text
/curated/?collection=model_runs
```

---

## 8. API Gateway

Documentation interactive :

```text
http://localhost:8000/docs
```

Endpoints :

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/health` | GET | Vérifie l'état de l'API et des connexions S3 / MySQL / MongoDB. |
| `/stats` | GET | Affiche les métriques du data lake et du pipeline ML. |
| `/raw/` | GET | Liste les objets du bucket raw. |
| `/staging/` | GET | Retourne les données staging depuis MySQL. |
| `/curated/` | GET | Retourne une collection curated MongoDB. |
| `/ingest` | POST | Endpoint d'ingestion naïf. |
| `/ingest_fast` | POST | Endpoint d'ingestion optimisé. |

Exemples :

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/stats | python3 -m json.tool
curl -s "http://localhost:8000/curated/?collection=model_runs&limit=5" | python3 -m json.tool
```

---

## 9. Niveau avancé : `/ingest` vs `/ingest_fast`

Le projet implémente deux endpoints d'ingestion comparables afin de répondre au niveau avancé du sujet.

| Endpoint | Implémentation | Objectif |
|---|---|---|
| `/ingest` | Traitement naïf, boucle Python et insertions ligne par ligne. | Servir de baseline simple et lisible. |
| `/ingest_fast` | Traitement optimisé, vectorisation NumPy / Numba et insertion batch. | Réduire le temps d'exécution, surtout lorsque le volume augmente. |

Le benchmark se lance avec :

```bash
python3 scripts/benchmark_ingest.py --runs 10 --batch-sizes 1 100
```

Ou depuis le conteneur API :

```bash
docker compose exec api python scripts/benchmark_ingest.py \
  --base-url http://localhost:8000 \
  --runs 10 \
  --batch-sizes 1 100
```

Les résultats sont écrits dans :

```text
data/benchmarks/ingest_benchmark_results.json
```

Lecture du fichier :

```bash
cat data/benchmarks/ingest_benchmark_results.json | python3 -m json.tool
```

Le gain est calculé ainsi :

```text
gain = (temps_ingest - temps_ingest_fast) / temps_ingest * 100
```

### 9.1 Résultats du benchmark local

Benchmark exécuté avec 10 runs par taille de batch :

```bash
python3 scripts/benchmark_ingest.py --runs 10 --batch-sizes 1 100
```

| Taille du batch | Moyenne `/ingest` | Moyenne `/ingest_fast` | Gain moyen | Gain médian | Objectif 30 % |
|---:|---:|---:|---:|---:|---|
| 1 | 0.0145 s | 0.1064 s | -632.61 % | -4.49 % | KO |
| 100 | 0.0362 s | 0.0186 s | +48.53 % | +49.53 % | OK |

Résultats détaillés :

| Batch | Run | `/ingest` | `/ingest_fast` |
|---:|---:|---:|---:|
| 1 | 1 | 0.0377 s | 0.9527 s |
| 1 | 2 | 0.0131 s | 0.0121 s |
| 1 | 3 | 0.0114 s | 0.0128 s |
| 1 | 4 | 0.0121 s | 0.0125 s |
| 1 | 5 | 0.0123 s | 0.0122 s |
| 1 | 6 | 0.0114 s | 0.0123 s |
| 1 | 7 | 0.0118 s | 0.0125 s |
| 1 | 8 | 0.0116 s | 0.0120 s |
| 1 | 9 | 0.0122 s | 0.0128 s |
| 1 | 10 | 0.0117 s | 0.0125 s |
| 100 | 1 | 0.0364 s | 0.0176 s |
| 100 | 2 | 0.0368 s | 0.0209 s |
| 100 | 3 | 0.0416 s | 0.0216 s |
| 100 | 4 | 0.0412 s | 0.0202 s |
| 100 | 5 | 0.0371 s | 0.0168 s |
| 100 | 6 | 0.0316 s | 0.0183 s |
| 100 | 7 | 0.0345 s | 0.0172 s |
| 100 | 8 | 0.0355 s | 0.0180 s |
| 100 | 9 | 0.0330 s | 0.0173 s |
| 100 | 10 | 0.0341 s | 0.0183 s |

### 9.2 Analyse des différences

Sur un batch de 100 éléments, `/ingest_fast` est nettement plus performant : le temps moyen passe de 0.0362 seconde à 0.0186 seconde, soit un gain moyen de 48.53 %. Ce résultat valide l'intérêt de l'optimisation dès que l'on traite un volume un peu plus significatif. Le gain vient principalement de deux choix techniques :

- les insertions sont regroupées au lieu d'être exécutées ligne par ligne ;
- les calculs sont vectorisés et préparés pour être traités efficacement par NumPy / Numba.

Sur un batch de 1 élément, le résultat est différent : `/ingest_fast` n'est pas plus rapide. Le premier appel est fortement pénalisé par un coût fixe de démarrage, notamment lié à l'initialisation / compilation JIT Numba. Même après ce premier run, l'optimisation apporte peu d'intérêt pour un seul élément, car le coût HTTP, la validation du payload, la connexion à la base et l'orchestration de la requête dominent le temps total.

Ce résultat est cohérent avec l'objectif réel de `/ingest_fast` : optimiser les traitements par lots. Pour un élément isolé, le pipeline naïf reste compétitif car il évite l'overhead de préparation. Pour un batch de 100 éléments, l'approche optimisée amortit ce coût fixe et devient presque deux fois plus rapide.

### 9.3 Conclusion benchmark

Le benchmark montre que :

```text
Batch 1   : /ingest_fast n'est pas avantageux à cause de l'overhead fixe.
Batch 100 : /ingest_fast respecte l'objectif avancé avec +48.53 % de gain moyen.
```

L'optimisation est donc pertinente en régime batch, ce qui correspond au cas d'usage principal d'un endpoint d'ingestion dans un data lake : absorber plusieurs lignes ou événements à la fois plutôt que traiter uniquement un élément isolé.

---
## 10. Machine Learning

Le modèle principal est un GRU entraîné sur des fenêtres temporelles de 30 jours.

Deux modes sont comparés :

| Mode | Features |
|---|---|
| `price_only` | rendement log, volatilité, RSI, ratio de moyennes mobiles |
| `full` | features prix + VIX + Fear & Greed |

Le pipeline ML est orchestré par Airflow via :

```text
market_mood_ml_pipeline
```

Les sorties sont stockées ici :

| Sortie | Emplacement |
|---|---|
| Dataset exporté | `data/curated_export/market_sequences.npz` |
| Artefacts JSON | `models/model_runs/*.json` |
| Runs ML MongoDB | `curated.model_runs` |
| Dernier run exposé | `/stats` |

Vérifier les artefacts :

```bash
ls -lh data/curated_export
ls -lh models/model_runs
```

Vérifier MongoDB :

```bash
docker compose exec mongodb mongosh --quiet --eval '
const dbx = db.getSiblingDB("curated");
printjson(
  dbx.model_runs.find(
    {},
    {_id:0, run_id:1, mode:1, metrics:1, created_at:1}
  ).sort({created_at:-1}).limit(5).toArray()
);
'
```

---

## 11. Résultats ML principaux

Les expériences montrent que le signal directionnel reste faible et instable.

Résumé méthodologique :

- split strictement temporel, jamais aléatoire ;
- comparaison systématique à une baseline de classe majoritaire ;
- évaluation par accuracy et AUC-ROC ;
- early stopping sur la validation loss ;
- validation walk-forward pour vérifier la stabilité temporelle.

Conclusion : aucun signal directionnel robuste ne se maintient de manière stable sur les différents découpages temporels. Ce résultat négatif est conservé et documenté, car il est plus crédible qu'un résultat positif non robuste.

---

## 12. DVC

Un pipeline DVC est disponible pour reproduire localement les étapes principales sans passer par Airflow :

```bash
dvc repro
```

Fichiers concernés :

```text
dvc.yaml
params.yaml
```

Airflow reste l'orchestrateur principal de démonstration.

---

## 13. Structure du projet

```text
market-mood-lake/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.api
├── Makefile
├── dvc.yaml
├── params.yaml
├── build/
│   ├── requirements.txt
│   ├── requirements-api.txt
│   ├── requirements-airflow.txt
│   ├── requirements-py314.txt
│   ├── unpack_to_raw.py
│   └── unpack_stocks_to_raw.py
├── dags/
│   ├── market_mood_pipeline.py
│   └── market_mood_ml_pipeline.py
├── data/
│   ├── kaggle_stocks/
│   │   └── SP500_Historical_Data.csv
│   ├── curated_export/
│   └── benchmarks/
├── models/
│   └── model_runs/
├── scripts/
│   ├── benchmark_ingest.py
│   ├── create_raw_bucket.py
│   └── smoke_api.py
├── src/
│   ├── ingestion/
│   │   └── market_api.py
│   ├── transform/
│   │   ├── preprocess_to_staging.py
│   │   ├── preprocess_to_staging_xs.py
│   │   ├── process_to_curated.py
│   │   └── process_to_curated_xs.py
│   ├── api/
│   │   ├── main.py
│   │   └── routes_ingest.py
│   └── ml/
│       ├── export_dataset.py
│       ├── train.py
│       ├── train_recorded.py
│       └── walk_forward.py
├── tests/
└── notebooks/
```

---

## 14. Smoke test / validation correction

Validation minimale à lancer avant rendu :

```bash
echo "===== DOCKER ====="
docker compose ps

echo "===== HEALTH ====="
curl -s http://localhost:8000/health | python3 -m json.tool

echo "===== STATS ====="
curl -s http://localhost:8000/stats | python3 -m json.tool

echo "===== RAW S3 ====="
docker compose exec localstack awslocal s3 ls s3://raw --recursive

echo "===== ML ARTIFACTS ====="
ls -lh data/curated_export || true
ls -lh models/model_runs || true

echo "===== CURATED MODEL RUNS ====="
curl -s "http://localhost:8000/curated/?collection=model_runs&limit=5" | python3 -m json.tool
```

Checklist finale :

```text
[OK] docker compose ps : services Up / healthy
[OK] /health : s3, mysql, mongodb à true
[OK] /stats : raw.object_count > 0
[OK] s3://raw contient sp500_combined.csv et market_mood_*.json
[OK] /staging/ retourne des lignes
[OK] /curated/?collection=model_runs retourne des runs ML
[OK] market_mood_pipeline est vert dans Airflow
[OK] market_mood_ml_pipeline est vert dans Airflow
[OK] models/model_runs contient des artefacts JSON
[OK] benchmark /ingest vs /ingest_fast généré
```

---

## 15. Dépannage rapide

### Airflow ne s'ouvre pas sur 8080

Dans ce projet, Airflow est exposé sur :

```text
http://localhost:8081
```

Vérifier :

```bash
docker compose ps
```

### `NoSuchBucket`

Le bucket `raw` n'existe pas encore. Le DAG `market_mood_pipeline` le crée automatiquement via `init_raw_sp500`. Sinon :

```bash
docker compose exec localstack awslocal s3 mb s3://raw
```

### `NoSuchKey: sp500_combined.csv`

Le fichier combiné n'est pas dans S3. Vérifier que le CSV source existe :

```bash
ls -lh data/kaggle_stocks/SP500_Historical_Data.csv
```

Puis relancer `market_mood_pipeline`.

### `/curated/` renvoie `[]`

Par défaut, `/curated/` lit `market_sequences`. Pour afficher les résultats ML :

```bash
curl -s "http://localhost:8000/curated/?collection=model_runs&limit=5" | python3 -m json.tool
```

### `python: command not found`

Sous WSL, utiliser souvent :

```bash
python3
```

au lieu de :

```bash
python
```

### Airflow ne recharge pas un DAG modifié

```bash
docker compose restart airflow-scheduler airflow-webserver
```

### LocalStack demande des credentials

Définir :

```bash
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
```

---

## 16. Conclusion

Market Mood Lake fournit un data lake complet, reproductible et démontrable :

- ingestion fichier + API ;
- stockage raw S3 LocalStack ;
- transformation staging MySQL ;
- exposition curated MongoDB / artefacts ML ;
- orchestration Airflow ;
- API Gateway FastAPI ;
- benchmark avancé `/ingest` vs `/ingest_fast` ;
- entraînement ML automatisé ;
- métriques ML stockées et consultables via `/stats`.

Le projet est conçu pour être lancé et vérifié rapidement par un correcteur : Docker démarre les services, Airflow orchestre le pipeline, FastAPI expose les résultats, et les métriques finales sont visibles dans MongoDB et via l'API.
