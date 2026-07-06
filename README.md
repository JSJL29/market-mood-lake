# Market Mood Lake

Data lake pour la prédiction de la direction (hausse/baisse) du S&P500, en
combinant l'historique de prix avec deux indicateurs de "sentiment" de
marché : le VIX et le Fear & Greed Index. Un GRU est entraîné sur des
séquences fenêtrées pour comparer la prédiction "prix seul" vs "prix +
sentiment".

Deux versions du modèle sont disponibles (voir section 3.5) :
- **Mono-indice** : le GRU est entraîné uniquement sur le S&P500.
- **Cross-sectional (coupe transversale)** : le même GRU est entraîné sur
  ~50 actions individuelles du S&P500 simultanément, ce qui multiplie le
  volume d'échantillons d'entraînement indépendants (~47x) et fait
  apparaître un signal directionnel faible mais réel, absent en
  mono-indice — conforme à ce que documente la littérature du domaine
  (voir section 3.5.2).

## 1. Architecture

Le data lake suit la convention **raw / staging / curated** vue en cours :

| Zone | Stockage | Contenu |
|---|---|---|
| Raw | S3 (LocalStack) | CSV S&P500 (Kaggle) + CSV multi-actions (Kaggle, ~472 tickers) + payloads JSON VIX/Fear&Greed horodatés |
| Staging | MySQL | Table `market_data` (indice unique) et table `market_data_xs` (multi-actions) : prix, volume, VIX, Fear&Greed alignés par date, features calculées (rendement log, volatilité glissante) |
| Curated | MongoDB | Collections `market_sequences` et `market_sequences_xs` : séquences fenêtrées (30 jours) + label hausse/baisse, prêtes pour l'entraînement |

Deux sources de données :
- **Fichier** : dataset Kaggle S&P500 historique (OHLCV).
- **API** : VIX via `yfinance`, Fear & Greed Index via l'API publique de CNN.

Orchestration : **Apache Airflow** (DAG `market_mood_pipeline`, rafraîchissement
toutes les 6h pour la partie API ; le CSV Kaggle est ingéré une seule fois
en amont). Un pipeline **DVC** équivalent est aussi disponible pour une
exécution reproductible en local.

API Gateway : **FastAPI**, avec les endpoints `/raw`, `/staging`,
`/curated`, `/health`, `/stats`, plus les endpoints avancés `/ingest` et
`/ingest_fast`.

## 2. Choix techniques

- **S3 (LocalStack) pour raw** : contrainte imposée par le sujet.
- **MySQL pour staging** : les données marché sont tabulaires et bien
  structurées (date, prix, indicateurs), un schéma SQL avec contrainte
  d'unicité sur la date est adapté pour la validation et la déduplication.
- **MongoDB pour curated** : chaque séquence fenêtrée est un document
  auto-porteur (features + label + métadonnées), pas besoin de schéma
  rigide, et c'est le format consommé directement par l'entraînement.
- **Numba (`@njit`)** pour le calcul vectorisé du rendement log et de la
  volatilité glissante sur tout l'historique en staging.
- **Split temporel (pas aléatoire)** pour le train/val/test du modèle :
  un split aléatoire mélangerait passé et futur et provoquerait une fuite
  d'information.
- **Export snapshot local (.npz) avant l'entraînement** plutôt qu'un
  DataLoader branché directement sur MongoDB : le volume de données est
  faible, et un snapshot évite une dépendance réseau à chaque epoch.

## 3. Installation

### Prérequis
- Docker Desktop, avec l'intégration WSL2 activée si vous êtes sous
  Windows (Docker Desktop → Settings → Resources → WSL Integration).
- **WSL2 fortement recommandé sous Windows** plutôt que PowerShell
  natif : Numba, torch et certaines dépendances installent plus
  difficilement sur Python très récent sous Windows, et les chemins
  Unix (`/tmp/`, etc.) posent problème hors WSL2.
- Python 3.10+. Sur Python 3.13/3.14, utilisez
  `build/requirements-py314.txt` (sans versions épinglées) plutôt que
  `build/requirements.txt`.
- **Le venv doit être créé en dehors du système de fichiers Windows
  monté** (pas sous `/mnt/c/...`) : `python3 -m venv ~/market-mood-lake-venv`
  plutôt que `.venv` dans le dossier projet. Installer pip dans un venv
  situé sous `/mnt/c/...` échoue avec des erreurs de permissions
  (`Operation not permitted`), car ce montage ne gère pas les
  permissions Unix comme un vrai système de fichiers Linux. Le code
  source, lui, peut rester sous `/mnt/c/...` sans problème.
- Un compte Kaggle (pour télécharger les datasets S&P500 et multi-actions).

### Étapes

```bash
git clone <repo_url>
cd market-mood-lake
cp .env.example .env

python3 -m venv ~/market-mood-lake-venv
source ~/market-mood-lake-venv/bin/activate
python -m pip install -r build/requirements.txt   # ou requirements-py314.txt
```

Lancez les services :

```bash
docker compose up -d --build
```

> Le premier build de l'image Airflow peut prendre du temps si les
> `requirements-airflow.txt` ne sont pas correctement allégés (voir
> section 5). Les credentials AWS factices (`AWS_ACCESS_KEY_ID=test`,
> etc.) sont déjà injectées dans le `docker-compose.yml` pour les
> services `api` et `airflow-*` ; boto3 les exige même pour parler à
> LocalStack, qui ne les vérifie jamais réellement.

Créez le bucket S3 nécessaire :

```bash
aws --endpoint-url=http://localhost:4566 s3 mb s3://raw
```

> Si `aws` n'est pas installé : `python -m pip install awscli`.
> LocalStack n'a pas de volume persistant par défaut au-delà de ce que
> `docker-compose.yml` déclare (`PERSISTENCE=1` + volume dédié) : sans
> ça, le bucket et son contenu sont perdus à chaque recréation du
> conteneur.

### 3.1 Lancer le pipeline mono-indice

```bash
python build/unpack_to_raw.py --input_dir data/kaggle_sp500 --bucket_name raw --output_file_name sp500_combined.csv

python src/ingestion/market_api.py --bucket_name raw --endpoint-url http://localhost:4566 --period max --start_date 2015-01-01

python src/transform/preprocess_to_staging.py --bucket_raw raw --sp500_file sp500_combined.csv --db_host localhost --db_user root --db_password root --endpoint-url http://localhost:4566 --horizon 5

python src/transform/process_to_curated.py --mysql_host localhost --mysql_user root --mysql_password root --mongo_uri mongodb://localhost:27017/ --window_size 30
```

**Option Airflow** (orchestration automatisée des 3 étapes VIX/F&G →
staging → curated ; le CSV Kaggle reste ingéré une seule fois
manuellement en amont, comme ci-dessus) :

1. Interface Airflow sur `localhost:8081` (identifiants : `airflow` / `airflow`).
2. Activez le DAG `market_mood_pipeline`, puis déclenchez un run manuel
   (bouton ▶️). Ne laissez jamais un DAG dont le `start_date` est très
   antérieur à aujourd'hui s'activer sans `catchup=False` (voir
   section 5) : Airflow rattraperait tous les créneaux passés d'un coup.

**Option DVC** (alternative locale) :

```bash
dvc repro
```

### 3.2 Lancer l'API

```bash
docker compose up -d api
```

ou manuellement (hors Docker) :

```bash
uvicorn src.api.main:app --reload
```

Documentation interactive : `http://localhost:8000/docs`

### 3.3 Benchmark `/ingest` vs `/ingest_fast`

Voir section 4.

### 3.4 Entraîner le modèle — mono-indice

```bash
python src/ml/export_dataset.py --mongo_uri mongodb://localhost:27017/ --output_path data/curated_export/market_sequences.npz
python -m src.ml.train --npz_path data/curated_export/market_sequences.npz --mode full        # prix + VIX + Fear&Greed
python -m src.ml.train --npz_path data/curated_export/market_sequences.npz --mode price_only  # prix seul, pour comparaison
```

> **Toujours lancer avec `python -m src.ml.train`** (et non
> `python src/ml/train.py`), depuis la racine du projet, pour que les
> imports `src.ml.*` se résolvent correctement. Même remarque pour tout
> script sous `src/` qui importe un autre module `src.*`
> (`preprocess_to_staging_xs.py` par exemple) : lancez-le avec
> `python -m src.transform.preprocess_to_staging_xs ...`.

**Résultats** (GRU, fenêtre de 30 jours, 2837 séquences 2015-2026, split
temporel 70/15/15, early stopping sur `val_loss`, horizon de label J+5) :

| Mode | Features | Baseline test (classe majoritaire) | Accuracy test | AUC-ROC test | Verdict |
|---|---|---|---|---|---|
| `price_only` | rendement log, volatilité glissante | 0.5892 | 0.5892 | 0.4488 | égal à la baseline, AUC < 0.5 |
| `full` | + VIX, Fear & Greed Index | 0.5892 | 0.5892 | 0.4584 | égal à la baseline, AUC < 0.5 |

Trois observations convergentes, obtenues avec des configurations
différentes (horizon J+1 puis J+5, accuracy puis AUC-ROC, avec et sans
early stopping), mènent à la même conclusion : le modèle converge
systématiquement vers la prédiction triviale de la classe majoritaire.
L'AUC-ROC, plus sensible que l'accuracy près de la baseline, confirme
l'absence de signal : une valeur proche ou inférieure à 0.5 indique que le
classement des probabilités produites par le modèle n'est pas meilleur
que le hasard, quelle que soit la configuration de features testée. Ni le
prix seul ni l'ajout du VIX / Fear & Greed Index n'apportent de signal
directionnel exploitable sur ce dataset.

Le niveau de prix brut a volontairement été exclu des features du modèle
(voir `src/ml/train.py`), car une série non-stationnaire introduirait des
corrélations parasites plutôt qu'un signal prédictif réel.

### 3.5 Entraîner le modèle — cross-sectional (coupe transversale)

**Mise en perspective avec la littérature.** Le résultat négatif
ci-dessus est cohérent avec les travaux publiés sur la prédiction
directionnelle d'indices boursiers. La référence du domaine, Fischer &
Krauss (2018), qui prédit la direction quotidienne des actions du S&P 500
sur la période 1992-2015, obtient une accuracy directionnelle à peine
supérieure au hasard (autour de 52-56 %) malgré un volume d'entraînement
des centaines de fois supérieur à celui du pipeline mono-indice. Ce
volume s'explique par une différence méthodologique clé : une approche en
**coupe transversale** entraîne un seul modèle sur des centaines de
titres simultanément (chaque titre apportant des exemples relativement
indépendants), alors que le pipeline mono-indice se limite à un seul
indice observé sur une seule période, ce qui borne mécaniquement le
nombre d'échantillons réellement indépendants.

Une revue de plusieurs projets similaires publiés sur GitHub confirme par
ailleurs un point important : la quasi-totalité prédisent le **prix**
(régression, métrique RMSE) plutôt que la **direction** (classification),
et aucun ne compare son résultat à une baseline chiffrée. Une prédiction
naïve du type "prix de demain ≈ prix d'aujourd'hui" obtient déjà un RMSE
très faible sur des séries de prix peu volatiles d'un jour à l'autre,
sans aucune utilité pratique pour une décision d'achat/vente — un piège
que ce projet évite en se concentrant sur la direction, avec une baseline
systématiquement affichée. Un de ces projets (`YC-Coder-Chen/GRU-stock-price-prediction`)
admet d'ailleurs lui-même, dans sa propre documentation, que son
architecture custom n'obtient qu'un résultat "légèrement meilleur" qu'une
prédiction naïve, et suppose que "les données de marché statiques ne
contiennent pas beaucoup d'information" — un constat indépendant qui
rejoint celui de ce projet.

Un second pipeline, parallèle au premier (le pipeline mono-indice reste
inchangé et fonctionnel), a donc été ajouté pour tester l'hypothèse
"coupe transversale" directement sur ce projet, avec deux extensions
supplémentaires : des indicateurs techniques (RSI, ratio de moyennes
mobiles) en plus du rendement/volatilité, et une passe à l'échelle
complète (472 tickers) :

```bash
# Ingestion multi-actions (--max_tickers 50 pour un premier test rapide, à omettre pour les 472 en entier)
python build/unpack_stocks_to_raw.py --input_dir data/kaggle_stocks --bucket_name raw --output_file_name stocks_combined.csv --max_tickers 50

# Staging : features calculées PAR TICKER séparément (jamais à cheval sur deux actions)
python -m src.transform.preprocess_to_staging_xs --bucket_raw raw --stocks_file stocks_combined.csv --db_host localhost --db_user root --db_password root --endpoint-url http://localhost:4566 --horizon 5

# Curated : fenêtrage PAR TICKER séparément (streaming, un ticker à la fois), regroupé dans une seule collection
python -m src.transform.process_to_curated_xs --mysql_host localhost --mysql_user root --mysql_password root --mongo_uri mongodb://localhost:27017/ --window_size 30

# Export et entraînement
python src/ml/export_dataset.py --mongo_uri mongodb://localhost:27017/ --output_path data/curated_export/market_sequences_xs.npz --collection market_sequences_xs
python -m src.ml.train --npz_path data/curated_export/market_sequences_xs.npz --mode full
python -m src.ml.train --npz_path data/curated_export/market_sequences_xs.npz --mode price_only
```

> À grande échelle (472 tickers, ~1,25M séquences), `process_to_curated_xs.py`
> et `export_dataset.py` traitent les données en flux (un ticker à la
> fois, tableaux NumPy préalloués) plutôt que de matérialiser toutes les
> séquences en mémoire d'un coup : l'overhead des objets Python sur
> plusieurs millions de séquences imbriquées peut sinon consommer
> plusieurs dizaines de Go de RAM. Vérifiez l'espace disponible
> (`free -h`) avant un run à pleine échelle.

**Résultats (50 tickers, 132 140 séquences, split unique 70/15/15) :**

| Mode | Baseline test | Accuracy test | AUC-ROC test | Verdict |
|---|---|---|---|---|
| `price_only` | 0.5340 | 0.5340 | 0.5105 | égal à la baseline, AUC proche de 0.5 |
| `full` | 0.5340 | **0.5355** | **0.5200** | **au-dessus** de la baseline, AUC > 0.5 |

Avec ~47x plus de séquences d'entraînement (132 140 contre 2837), le
modèle `full` dépassait pour la première fois la baseline sur les deux
métriques. Ce résultat, obtenu sur un split unique, a motivé deux essais
supplémentaires pour tenter de l'améliorer : l'ajout d'indicateurs
techniques (RSI, ratio de moyennes mobiles court/long terme) et un passage
à l'échelle complète (472 tickers, 1 248 697 séquences). Résultat de ces
deux essais combinés, même après plusieurs configurations d'hyperparamètres
(hidden_size 32 à 64, 1 à 2 couches, dropout 0.2 à 0.3, scheduler de
learning rate) :

| Config | Mode | Baseline test | Accuracy test | AUC-ROC test |
|---|---|---|---|---|
| 472 tickers + indicateurs techniques | `full` | 0.5329 | 0.5038–0.5280 | 0.4988–0.5207 |
| 472 tickers + indicateurs techniques | `price_only` | 0.5329 | 0.5325–0.5333 | 0.4988–0.5062 |

Le modèle `full` à 472 tickers **surapprend nettement** quelle que soit la
capacité testée (`train_loss` qui chute pendant que `val_loss` augmente
dès les premières epochs), et son résultat final ne dépasse plus la
baseline — contrairement au résultat observé à 50 tickers. Ce
renversement, obtenu en changeant seulement l'échelle et les
hyperparamètres sans changer la méthode de split, a motivé un test de
robustesse plus poussé avant de conclure quoi que ce soit : le résultat
positif à 50 tickers était-il un vrai signal, ou un artefact du split
unique utilisé pour l'évaluer ?

**Validation walk-forward : le test décisif.** Un split unique
train/val/test peut donner un résultat favorable par pur hasard de
découpage, sans que le modèle ait appris quoi que ce soit de
généralisable. La validation walk-forward découpe la chronologie en
plusieurs segments successifs et répète l'entraînement/évaluation sur
chacun : un signal réel doit apparaître de façon cohérente sur tous les
segments, pas seulement sur un seul découpage favorable.

```bash
python -m src.ml.walk_forward --npz_path data/curated_export/market_sequences_xs.npz --mode full --n_folds 5 --hidden_size 32 --num_layers 1 --dropout 0.2
python -m src.ml.walk_forward --npz_path data/curated_export/market_sequences_xs.npz --mode price_only --n_folds 5 --hidden_size 32 --num_layers 1 --dropout 0.2
```

**Résultats (50 tickers, 5 segments temporels successifs) :**

| Mode | Écart moyen accuracy − baseline | AUC moyenne | Signe de l'écart cohérent sur tous les segments |
|---|---|---|---|
| `full` | **−0.0219** | 0.5071 | **OUI — systématiquement en dessous de la baseline** |
| `price_only` | −0.0087 | 0.5031 | NON — change de signe d'un segment à l'autre |

Ce test invalide le résultat positif observé sur le split unique à 50
tickers : sur les 4 segments temporels testés, `full` est **chaque fois**
en dessous de la baseline, ce qui indique que le résultat favorable
obtenu précédemment était dû au hasard du découpage plutôt qu'à un signal
stable et exploitable. Le cas de `price_only`, où le signe de l'écart
change d'un segment à l'autre, est la signature d'une **dérive de
régime** ("concept drift") : la relation entre les indicateurs
techniques et la direction future du marché n'est pas stable dans le
temps, elle dépend de la période observée.

**Conclusion finale.** Sur l'ensemble des configurations testées dans ce
projet — mono-indice et cross-sectional (50 puis 472 tickers), horizon
J+1 et J+5, avec et sans indicateurs techniques, plusieurs architectures
et hyperparamètres, évaluées par accuracy et par AUC-ROC — aucune ne
produit de signal directionnel robuste et stable dans le temps une fois
soumise à une validation walk-forward. Les résultats ponctuellement
positifs obtenus sur un split unique ne résistent pas à une validation
sur plusieurs segments temporels successifs, ce qui indique qu'ils
étaient dus au hasard du découpage plutôt qu'à un véritable pouvoir
prédictif. Ce résultat négatif, obtenu par une méthodologie qui inclut
explicitement son propre test de robustesse, est cohérent avec
l'hypothèse d'efficience des marchés et avec la littérature académique
citée plus haut.

## 4. Niveau avancé : `/ingest` vs `/ingest_fast`

`/ingest` calcule les features avec une boucle Python pure et insère les
lignes une par une dans MySQL. `/ingest_fast` vectorise le calcul avec
NumPy, le compile avec Numba (`@njit`), et insère par batch
(`executemany`).

Pour reproduire le benchmark :

```bash
python scripts/benchmark_ingest.py --api_url http://localhost:8000
```

Résultats de benchmark (moyenne sur 5 essais, warm-up Numba exclu ;
mesuré via `scripts/benchmark_ingest.py`) :

| Endpoint | Batch=1 | Batch=100 |
|---|---|---|
| `/ingest` | 0.019055 s | 0.028891 s |
| `/ingest_fast` | 0.010250 s | 0.015310 s |
| **Gain** | **46.2 %** | **47.0 %** |

Le gain dépasse le seuil de 30 % demandé, sur les deux tailles de batch.
À cette échelle (1 à 100 lignes), le gain provient principalement de
l'insertion MySQL par batch (`executemany` au lieu d'un `INSERT` par
ligne) plutôt que de la vectorisation Numba elle-même : la compilation
JIT a un coût fixe au premier appel (~1.6 s), largement amorti seulement
à partir de volumes bien plus importants. C'est pourquoi le warm-up est
mesuré séparément et exclu de la moyenne : il reflète un coût de
démarrage, pas la performance en régime établi.

## 5. Difficultés rencontrées

Quelques incidents rencontrés pendant le développement, gardés ici car
ils illustrent des pièges réels de mise en production d'un data lake, et
pour éviter à quelqu'un d'autre de perdre du temps dessus :

**Environnement Windows / WSL2**
- **`wsl` lançait la distro interne `docker-desktop`** au lieu d'une
  vraie distro Ubuntu (repérable au prompt `docker-desktop:/tmp/...#`
  et à l'absence de `/mnt/c`). Fix : `wsl --install -d Ubuntu` puis
  `wsl --set-default Ubuntu`.
- **`python3 -m venv` échoue avec `ensurepip` (`exit status 1`)** sur
  certaines installations Ubuntu récentes. Fix : créer le venv avec
  `--without-pip`, puis installer pip manuellement via
  `curl -sS https://bootstrap.pypa.io/get-pip.py | python3` — **sans
  jamais utiliser `sudo`** dans cette commande (voir point suivant).
- **Un venv situé sous `/mnt/c/...` (donc sur un disque Windows monté)
  échoue à l'installation de pip** avec `OSError: [Errno 1] Operation
  not permitted`, avec ou sans `sudo` : ce montage ne respecte pas les
  permissions Unix comme un vrai filesystem Linux. Fix : créer le venv
  sous `~/` (filesystem Linux natif), garder uniquement le code source
  sous `/mnt/c/...`.
- **`aws` / `docker compose` "not found"** après un copier-coller de
  plusieurs commandes d'un coup : certains terminaux n'exécutent pas
  chaque ligne d'un bloc collé comme des commandes séparées. Toujours
  valider une commande à la fois avec Entrée pour les étapes sensibles.
- **`ModuleNotFoundError: No module named 'src'`** en lançant un script
  directement (`python src/ml/train.py`) : toujours utiliser
  `python -m src.chemin.vers.module` depuis la racine du projet pour
  tout script qui importe un autre module `src.*`.
- **`python: command not found` / `NoCredentialsError` réapparus dans un
  nouveau terminal** : le venv activé (`source ~/.../bin/activate`) et
  les variables d'environnement (`export AWS_ACCESS_KEY_ID=...`) ne
  persistent que pour la session de terminal où ils ont été définis.
  Chaque nouvel onglet/fenêtre de terminal doit refaire les deux, ou
  les ajouter à `~/.bashrc`.

**Docker Desktop**
- **Plantages `wsl-bootstrap` répétés** (`WSL integration with distro
  'Ubuntu' unexpectedly stopped`, parfois avec un `stack overflow` /
  `exit status 0xc00000fd`) : redémarrage de l'intégration WSL depuis
  Docker Desktop, ou en dernier recours `wsl --shutdown` +
  redémarrage complet de Docker Desktop.
- **`failed to stat parent: ... no such file or directory`** lors d'un
  build après une instabilité prolongée : signe de corruption du
  stockage interne de containerd. Fix : `docker system prune -a --volumes`
  puis rebuild complet.
- **Build Airflow en parallèle sur 3 services** (`airflow-webserver`,
  `airflow-scheduler`, `airflow-init` avaient chacun `build: .`) :
  collision de tag d'image (`already exists`, `CANCELED`). Fix : un seul
  service (`airflow-init`) construit l'image, les deux autres la
  réutilisent via `depends_on: condition: service_completed_successfully`.

**LocalStack et credentials**
- **LocalStack a changé de modèle de licence en mars 2026** : l'image
  `localstack/localstack:latest` exige désormais un compte et un token
  d'authentification, même pour émuler S3 en local. Le `docker-compose.yml`
  épingle donc explicitement `localstack:2.3` (dernière version stable
  avant ce changement). Leçon retenue : ne jamais utiliser `:latest` pour
  une dépendance critique dans un projet destiné à être reproductible.
- **`NoCredentialsError` / `Unable to locate credentials`** : boto3 exige
  des credentials AWS même pour parler à LocalStack (qui ne les vérifie
  jamais). Fix : variables d'environnement factices
  (`AWS_ACCESS_KEY_ID=test`, etc.), déclarées à la fois dans le shell et
  dans le `docker-compose.yml` pour les services `api`/`airflow-*` — les
  définir dans un seul des deux endroits ne suffit pas si le script
  tourne aussi dans l'autre contexte.
- **Bucket perdu à chaque redémarrage de conteneur** : LocalStack n'a
  pas de volume persistant par défaut. Fix : `PERSISTENCE=1` +
  volume Docker dédié dans `docker-compose.yml`.

**Airflow**
- **Incompatibilité Python dans le conteneur Airflow** : l'image
  `apache/airflow:2.7.1` tourne sur Python 3.8, mais installer `yfinance`
  sans épingler ses dépendances a récupéré une version de `multitasking`
  utilisant une syntaxe (`type[Thread]`) qui nécessite Python 3.9+. Fix :
  épingler `multitasking==0.0.11` dans `build/requirements-airflow.txt`.
- **`catchup` d'Airflow** : un premier `start_date` daté de plusieurs mois
  dans le passé a fait "rattraper" une quinzaine d'exécutions d'un coup
  dès l'activation du DAG, saturant les ressources. Fix : `catchup=False`
  et `max_active_runs=1`.
- **Chemins de scripts dans le conteneur Airflow** : le volume
  `./src:/opt/airflow/scripts` monte l'arborescence complète de `src/`
  (avec ses sous-dossiers `ingestion/`, `transform/`, etc.), ce qui
  écrase la copie à plat faite par le `Dockerfile`. Les `BashOperator`
  du DAG doivent donc référencer les scripts avec leur sous-dossier
  (`scripts/ingestion/market_api.py`), pas à la racine de `scripts/`.
- **Un `dags/*.py` modifié localement met jusqu'à quelques minutes à
  être rechargé** par le scheduler Airflow. Pour forcer un rechargement
  immédiat après un changement de DAG : `docker restart airflow-scheduler`.

**Sources de données**
- **Dataset Kaggle S&P500 initial arrêté en novembre 2020** : repéré via
  la validation `MAX(date)` en staging, remplacé par un dataset couvrant
  jusqu'à début juillet 2026.
- **CNN Fear & Greed** : l'endpoint live de CNN bloque les requêtes sans
  User-Agent réaliste (418) et renvoie des erreurs serveur sur de grandes
  plages de dates (500). Remplacé par une copie historique maintenue sur
  GitHub (`whit3rabbit/fear-greed-data`), mise à jour quotidiennement et
  couvrant 2011 à aujourd'hui, plus robuste pour un usage reproductible.

**Modélisation (cross-sectional à grande échelle)**
- **Mémoire** : construire la liste complète des séquences (documents
  MongoDB imbriqués) pour l'ensemble des tickers avant la moindre
  insertion peut consommer plusieurs dizaines de Go à l'échelle de
  472 tickers (~1,25M séquences). Fix : traitement en flux, un ticker à
  la fois (`process_to_curated_xs.py`), et préallocation des tableaux
  NumPy lors de l'export plutôt que matérialisation de tous les
  documents en liste Python (`export_dataset.py`).
- **Un résultat positif sur un split unique n'est pas une preuve de
  signal réel** : passer de 50 à 472 tickers a fait apparaître un net
  surapprentissage (quelle que soit la capacité du modèle testée), et un
  test de validation walk-forward (plusieurs segments temporels
  successifs) a révélé que le résultat positif observé à 50 tickers ne
  se répétait pas de façon cohérente dans le temps — signe qu'il était dû
  au hasard du découpage plutôt qu'à un vrai pouvoir prédictif. Toujours
  valider un résultat favorable sur plusieurs découpages temporels avant
  de le considérer comme acquis, en particulier sur des séries
  financières où la relation entre variables peut ne pas être stable
  dans le temps ("concept drift").

## 6. Structure du projet

```
market-mood-lake/
├── docker-compose.yml
├── Dockerfile              # image Airflow
├── Dockerfile.api          # image FastAPI
├── dvc.yaml / params.yaml
├── build/
│   ├── requirements.txt          # venv local complet (avec torch)
│   ├── requirements-py314.txt    # idem, sans versions épinglées (Python 3.13+)
│   ├── requirements-airflow.txt  # conteneur Airflow (allégé)
│   ├── requirements-api.txt      # conteneur API (allégé)
│   ├── unpack_to_raw.py          # CSV S&P500 (indice) -> raw S3
│   └── unpack_stocks_to_raw.py   # CSV multi-actions -> raw S3
├── dags/
│   └── market_mood_pipeline.py
├── scripts/
│   └── benchmark_ingest.py     # benchmark /ingest vs /ingest_fast
├── src/
│   ├── ingestion/
│   │   └── market_api.py       # VIX + Fear&Greed -> raw S3
│   ├── transform/
│   │   ├── preprocess_to_staging.py      # raw -> MySQL (indice unique)
│   │   ├── preprocess_to_staging_xs.py   # raw -> MySQL (cross-sectional)
│   │   ├── process_to_curated.py         # MySQL -> MongoDB (indice unique)
│   │   └── process_to_curated_xs.py      # MySQL -> MongoDB (cross-sectional)
│   ├── api/
│   │   ├── main.py             # FastAPI (/raw /staging /curated /health /stats)
│   │   └── routes_ingest.py    # /ingest /ingest_fast
│   └── ml/
│       ├── export_dataset.py   # --collection market_sequences | market_sequences_xs
│       ├── dataset.py
│       ├── train.py
│       └── walk_forward.py     # validation walk-forward (robustesse temporelle)
└── notebooks/
    └── exploration.ipynb
```
