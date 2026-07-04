import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import mysql.connector
from pymongo import MongoClient
from numpy.lib.stride_tricks import sliding_window_view


FEATURE_COLUMNS = ["close", "log_return", "rolling_vol_20d", "rsi_14", "ma_ratio", "vix_close", "fear_greed_score"]


def get_mysql_data(host, user, password, database):
    """Récupère les données depuis MySQL, ordonnées chronologiquement."""
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        query = "SELECT * FROM market_data ORDER BY date ASC"
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données MySQL: {e}")
        return None


def build_sequences(df, window_size=30):
    """
    Construit des fenêtres glissantes de `window_size` jours, de façon
    vectorisée (NumPy `sliding_window_view`) plutôt qu'avec une boucle
    Python : indispensable pour rester rapide sur de gros volumes
    (des centaines de milliers à quelques millions de lignes en
    cross-sectional).
    """
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up"]).reset_index(drop=True)
    n = len(df)
    if n <= window_size:
        return []

    feature_matrix = df[FEATURE_COLUMNS].astype(float).values  # (n, n_features)

    # windows[k] = feature_matrix[k : k+window_size]
    windows = sliding_window_view(feature_matrix, window_size, axis=0)  # (n-window_size+1, n_features, window_size)
    windows = windows.transpose(0, 2, 1)  # (n-window_size+1, window_size, n_features)
    windows = windows[:-1]  # on exclut la dernière fenêtre (pas de label après elle)

    labels = df["target_up"].values[window_size:n].astype(int)
    dates = df["date"].values[window_size:n]

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
    try:
        client = MongoClient(mongo_uri)
        db = client.curated
        collection = db.market_sequences

        collection.delete_many({})
        result = collection.insert_many(documents)
        print(f"Nombre de documents insérés: {len(result.inserted_ids)}")

        print("\nExemple de documents insérés:")
        for doc in collection.find().limit(2):
            print(f"\nWindow end date: {doc['window_end_date']}")
            print(f"Label (hausse=1/baisse=0): {doc['label']}")
            print(f"Shape de la fenêtre: {len(doc['features'])}x{len(doc['features'][0])}")

        client.close()
        return True
    except Exception as e:
        print(f"Erreur lors de l'insertion dans MongoDB: {e}")
        return False


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

    if df is None or df.empty:
        print("Aucune donnée récupérée depuis MySQL")
        return

    print(f"Nombre de lignes récupérées: {len(df)}")

    print(f"\nConstruction des séquences (fenêtre = {args.window_size} jours)...")
    sequences = build_sequences(df, window_size=args.window_size)
    print(f"Nombre de séquences construites: {len(sequences)}")

    print("\nPréparation des documents pour MongoDB...")
    documents = prepare_mongodb_documents(sequences, args.window_size)

    print("\nInsertion des documents dans MongoDB...")
    success = insert_to_mongodb(documents, args.mongo_uri)

    if success:
        print("\nTraitement terminé avec succès!")
    else:
        print("\nErreur lors du traitement")


if __name__ == "__main__":
    main()