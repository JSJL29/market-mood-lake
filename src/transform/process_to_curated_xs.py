"""
Version cross-sectional du curated : construit les fenêtres glissantes
SÉPARÉMENT pour chaque ticker (jamais une fenêtre à cheval sur deux
actions différentes), puis regroupe toutes les séquences de tous les
tickers dans une seule collection MongoDB. C'est ce regroupement qui
donne le volume d'entraînement "coupe transversale".

Traitement en streaming (un ticker à la fois, insertion immédiate) :
à grande échelle (plusieurs centaines de tickers, plusieurs millions de
séquences), construire la liste complète de toutes les séquences en
mémoire avant la moindre insertion peut consommer plusieurs dizaines de
Go (overhead des objets Python). En traitant et en insérant ticker par
ticker, le pic mémoire reste borné à la taille d'un seul ticker.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import mysql.connector
from pymongo import MongoClient
from numpy.lib.stride_tricks import sliding_window_view

FEATURE_COLUMNS = ["close", "log_return", "rolling_vol_20d", "rsi_14", "ma_ratio", "vix_close", "fear_greed_score"]


def get_mysql_data_xs(host, user, password, database):
    """Récupère les données cross-sectional depuis MySQL, triées par ticker puis date."""
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        query = "SELECT * FROM market_data_xs ORDER BY ticker ASC, date ASC"
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données MySQL: {e}")
        return None


def _windows_for_group(feature_matrix, window_size):
    """Fenêtrage vectorisé pour un seul ticker."""
    n = len(feature_matrix)
    if n <= window_size:
        return np.empty((0, window_size, feature_matrix.shape[1]))
    windows = sliding_window_view(feature_matrix, window_size, axis=0)
    windows = windows.transpose(0, 2, 1)
    return windows[:-1]


def build_documents_for_ticker(ticker, group, window_size):
    """Construit les documents MongoDB pour un seul ticker (fenêtrage vectorisé)."""
    group = group.sort_values("date").reset_index(drop=True)
    n = len(group)
    if n <= window_size:
        return []

    feature_matrix = group[FEATURE_COLUMNS].astype(np.float32).values
    windows = _windows_for_group(feature_matrix, window_size)
    if len(windows) == 0:
        return []

    labels = group["target_up"].values[window_size:n].astype(int)
    dates = group["date"].values[window_size:n]
    processed_at = datetime.now(timezone.utc).isoformat()

    documents = [
        {
            "ticker": ticker,
            "window_end_date": str(dates[k]),
            "features": windows[k].tolist(),
            "label": int(labels[k]),
            "metadata": {
                "source": "mysql_staging_xs",
                "window_size": window_size,
                "feature_columns": FEATURE_COLUMNS,
                "processed_at": processed_at,
            },
        }
        for k in range(len(windows))
    ]
    return documents


def process_all_tickers_streaming(df, window_size, mongo_uri, insert_batch_size=5000):
    """
    Traite chaque ticker un par un : construit ses documents, les insère
    immédiatement dans MongoDB, puis libère la mémoire avant de passer
    au ticker suivant. Le pic mémoire reste borné à un seul ticker à la
    fois plutôt qu'à l'ensemble du dataset.
    """
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up"]).reset_index(drop=True)

    client = MongoClient(mongo_uri)
    collection = client.curated.market_sequences_xs
    collection.delete_many({})

    tickers = df["ticker"].unique()
    total_inserted = 0
    example_docs = []

    for idx, ticker in enumerate(tickers, start=1):
        group = df[df["ticker"] == ticker]
        documents = build_documents_for_ticker(ticker, group, window_size)

        if documents:
            for i in range(0, len(documents), insert_batch_size):
                batch = documents[i:i + insert_batch_size]
                result = collection.insert_many(batch)
                total_inserted += len(result.inserted_ids)
            if len(example_docs) < 2:
                example_docs.extend(documents[:2 - len(example_docs)])

        if idx % 25 == 0 or idx == len(tickers):
            print(f"  Ticker {idx}/{len(tickers)} ({ticker}) traité — {total_inserted} documents insérés au total")

        # Libère explicitement la référence pour aider le garbage collector
        # avant de passer au ticker suivant.
        del documents, group

    print(f"\nNombre total de documents insérés: {total_inserted}")

    print("\nExemple de documents insérés:")
    for doc in example_docs:
        print(f"\nTicker: {doc['ticker']} - Window end date: {doc['window_end_date']}")
        print(f"Label (hausse=1/baisse=0): {doc['label']}")
        print(f"Shape de la fenêtre: {len(doc['features'])}x{len(doc['features'][0])}")

    client.close()
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description='Traitement cross-sectional des données de staging vers curated')
    parser.add_argument('--mysql_host', type=str, default='localhost', help='Hôte MySQL')
    parser.add_argument('--mysql_user', type=str, default='root', help='Utilisateur MySQL')
    parser.add_argument('--mysql_password', type=str, default='root', help='Mot de passe MySQL')
    parser.add_argument('--mongo_uri', type=str, default='mongodb://localhost:27017/', help='URI MongoDB')
    parser.add_argument('--window_size', type=int, default=30, help="Taille de la fenêtre glissante (jours)")
    args = parser.parse_args()

    print("Récupération des données depuis MySQL (market_data_xs)...")
    df = get_mysql_data_xs(args.mysql_host, args.mysql_user, args.mysql_password, 'staging')

    if df is None or df.empty:
        print("Aucune donnée récupérée depuis MySQL")
        return

    print(f"Nombre de lignes récupérées: {len(df)} ({df['ticker'].nunique()} tickers)")
    print(f"\nConstruction et insertion des séquences par ticker (streaming, fenêtre = {args.window_size} jours)...")

    total = process_all_tickers_streaming(df, args.window_size, args.mongo_uri)

    if total > 0:
        print("\nTraitement terminé avec succès!")
    else:
        print("\nErreur lors du traitement (aucun document inséré)")


if __name__ == "__main__":
    main()