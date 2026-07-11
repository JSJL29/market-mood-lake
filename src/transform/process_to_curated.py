import argparse
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import mysql.connector
from pymongo import MongoClient
from numpy.lib.stride_tricks import sliding_window_view


FEATURE_COLUMNS = ["close", "log_return", "rolling_vol_20d", "rsi_14", "ma_ratio", "vix_close", "fear_greed_score"]


def valid_window_mask(dates, window_size, max_gap_days=7):
    if window_size < 1:
        raise ValueError("window_size doit être supérieur ou égal à 1")
    n = len(dates)
    if n < window_size:
        return np.zeros(0, dtype=bool)
    if window_size == 1:
        return np.ones(n, dtype=bool)
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    if parsed.isna().any():
        raise ValueError("Dates invalides dans les données Staging")
    gaps = parsed.diff().dt.days.to_numpy()[1:]
    gap_windows = sliding_window_view(gaps, window_size - 1)
    return np.max(gap_windows, axis=1) <= max_gap_days


def get_mysql_data(host, user, password, database):
    """Récupère les données depuis MySQL, ordonnées chronologiquement."""
    connection = None
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        query = "SELECT * FROM market_data ORDER BY date ASC"
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la récupération des données MySQL: {e}") from e
    finally:
        if connection is not None and connection.is_connected():
            connection.close()


def build_sequences(df, window_size=30):
    """
    Construit des fenêtres glissantes de `window_size` jours, de façon
    vectorisée (NumPy `sliding_window_view`) plutôt qu'avec une boucle
    Python : indispensable pour rester rapide sur de gros volumes
    (des centaines de milliers à quelques millions de lignes en
    cross-sectional).
    """
    required = FEATURE_COLUMNS + ["target_up", "date"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes de market_data: {missing}")
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up"]).reset_index(drop=True)
    n = len(df)
    if n < window_size:
        return []

    feature_matrix = df[FEATURE_COLUMNS].astype(float).values  # (n, n_features)

    # windows[k] = feature_matrix[k : k+window_size]
    windows = sliding_window_view(feature_matrix, window_size, axis=0)  # (n-window_size+1, n_features, window_size)
    windows = windows.transpose(0, 2, 1)  # (n-window_size+1, window_size, n_features)
    # target_up est déjà le label futur calculé pour chaque ligne Staging.
    # La fenêtre [0:window_size] se termine donc à l'index window_size-1
    # et doit recevoir le label de cette même ligne, pas celui du lendemain.
    labels = df["target_up"].values[window_size - 1:n].astype(int)
    dates = df["date"].values[window_size - 1:n]
    valid = valid_window_mask(df["date"].values, window_size)
    windows, labels, dates = windows[valid], labels[valid], dates[valid]

    sequences = [
        {
            "window_end_date": str(dates[k]),
            "features": windows[k].tolist(),
            "label": int(labels[k]),
        }
        for k in range(len(windows))
    ]
    return sequences


def prepare_mongodb_documents(sequences, window_size):
    """Prépare les documents pour MongoDB, avec métadonnées comme dans le TP3/TP6."""
    documents = []
    for seq in sequences:
        documents.append({
            "window_end_date": seq["window_end_date"],
            "features": seq["features"],
            "label": seq["label"],
            "metadata": {
                "source": "mysql_staging",
                "window_size": window_size,
                "feature_columns": FEATURE_COLUMNS,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        })
    return documents


def insert_to_mongodb(documents, mongo_uri="mongodb://localhost:27017/"):
    """Insère les documents fenêtrés dans MongoDB."""
    if not documents:
        raise ValueError("Aucun document Curated à insérer")
    client = None
    temporary_name = f"market_sequences__tmp_{uuid.uuid4().hex}"
    try:
        client = MongoClient(mongo_uri)
        db = client.curated
        collection = db[temporary_name]
        inserted = 0
        for start in range(0, len(documents), 1000):
            result = collection.insert_many(documents[start:start + 1000])
            inserted += len(result.inserted_ids)
        collection.create_index("window_end_date")
        if inserted != len(documents) or collection.count_documents({}) != len(documents):
            raise RuntimeError("Validation du chargement MongoDB temporaire échouée")
        collection.rename("market_sequences", dropTarget=True)
        collection = db.market_sequences
        print(f"Nombre de documents insérés: {inserted}")

        print("\nExemple de documents insérés:")
        for doc in collection.find().limit(2):
            print(f"\nWindow end date: {doc['window_end_date']}")
            print(f"Label (hausse=1/baisse=0): {doc['label']}")
            print(f"Shape de la fenêtre: {len(doc['features'])}x{len(doc['features'][0])}")

        return inserted
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'insertion dans MongoDB: {e}") from e
    finally:
        if client is not None:
            try:
                if temporary_name in client.curated.list_collection_names():
                    client.curated[temporary_name].drop()
            finally:
                client.close()


def main():
    parser = argparse.ArgumentParser(description='Traitement des données de staging vers curated')
    parser.add_argument('--mysql_host', type=str, default='localhost', help='Hôte MySQL')
    parser.add_argument('--mysql_user', type=str, default='root', help='Utilisateur MySQL')
    parser.add_argument('--mysql_password', type=str, default='root', help='Mot de passe MySQL')
    parser.add_argument('--mongo_uri', type=str, default='mongodb://localhost:27017/', help='URI MongoDB')
    parser.add_argument('--window_size', type=int, default=30, help="Taille de la fenêtre glissante (jours)")
    args = parser.parse_args()

    print("Récupération des données depuis MySQL...")
    df = get_mysql_data(args.mysql_host, args.mysql_user, args.mysql_password, 'staging')

    if df.empty:
        raise RuntimeError("Aucune donnée récupérée depuis MySQL")

    print(f"Nombre de lignes récupérées: {len(df)}")

    print(f"\nConstruction des séquences (fenêtre = {args.window_size} jours)...")
    sequences = build_sequences(df, window_size=args.window_size)
    print(f"Nombre de séquences construites: {len(sequences)}")
    if not sequences:
        non_null = {column: int(df[column].notna().sum()) for column in FEATURE_COLUMNS + ["target_up"]}
        raise RuntimeError(
            f"Aucune séquence construite après suppression des valeurs nulles; comptes non nuls: {non_null}"
        )

    print("\nPréparation des documents pour MongoDB...")
    documents = prepare_mongodb_documents(sequences, args.window_size)

    print("\nInsertion des documents dans MongoDB...")
    inserted = insert_to_mongodb(documents, args.mongo_uri)
    print(f"\nTraitement terminé avec succès: {inserted} documents insérés")


if __name__ == "__main__":
    main()
