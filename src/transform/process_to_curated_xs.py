"""
Version cross-sectional du curated avec features sectorielles.

Construit les fenêtres glissantes séparément pour chaque ticker, jamais une
fenêtre à cheval sur deux actions différentes, puis regroupe toutes les
séquences dans MongoDB curated.market_sequences_xs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import mysql.connector
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pymongo import MongoClient


# Ordre OFFICIEL des features stockées dans chaque document MongoDB.
# train.py et train_xgb_gru_attention.py doivent utiliser exactement le même ordre.
FEATURE_COLUMNS = [
    "close",
    "log_return",
    "rolling_vol_20d",
    "rsi_14",
    "ma_ratio",
    "ticker_momentum_5d",
    "vix_close",
    "fear_greed_score",
    "sector_log_return",
    "sector_rolling_vol_20d",
    "sector_momentum_5d",
    "ticker_minus_sector_log_return",
    "ticker_minus_sector_momentum_5d",
]


def get_mysql_data_xs(host: str, user: str, password: str, database: str) -> pd.DataFrame | None:
    """Récupère les données cross-sectional depuis MySQL, triées ticker/date."""
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        query = "SELECT * FROM market_data_xs ORDER BY ticker ASC, date ASC"
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données MySQL: {e}")
        return None


def _windows_for_group(feature_matrix: np.ndarray, window_size: int) -> np.ndarray:
    """Fenêtrage vectorisé pour un seul ticker."""
    n = len(feature_matrix)
    if n <= window_size:
        return np.empty((0, window_size, feature_matrix.shape[1]), dtype=np.float32)
    windows = sliding_window_view(feature_matrix, window_size, axis=0)
    windows = windows.transpose(0, 2, 1)
    return windows[:-1]


def build_documents_for_ticker(ticker: str, group: pd.DataFrame, window_size: int) -> list[dict]:
    """Construit les documents MongoDB pour un seul ticker."""
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

    documents = []
    for k in range(len(windows)):
        row = group.iloc[window_size + k]
        documents.append(
            {
                "ticker": ticker,
                "window_end_date": str(dates[k]),
                "features": windows[k].tolist(),
                "label": int(labels[k]),
                "metadata": {
                    "source": "mysql_staging_xs",
                    "window_size": window_size,
                    "feature_columns": FEATURE_COLUMNS,
                    "sector": row.get("sector", None),
                    "sector_etf": row.get("sector_etf", None),
                    "processed_at": processed_at,
                },
            }
        )
    return documents


def process_all_tickers_streaming(
    df: pd.DataFrame,
    window_size: int,
    mongo_uri: str,
    insert_batch_size: int = 5000,
) -> int:
    """Traite ticker par ticker pour garder un pic mémoire borné."""
    missing_cols = [col for col in FEATURE_COLUMNS + ["target_up"] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Colonnes absentes de market_data_xs : "
            f"{missing_cols}. Relance preprocess_to_staging_xs.py avec la version sectorielle."
        )

    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up"]).reset_index(drop=True)
    client = MongoClient(mongo_uri)
    collection = client.curated.market_sequences_xs
    collection.delete_many({})

    tickers = df["ticker"].unique()
    total_inserted = 0
    example_docs: list[dict] = []

    for idx, ticker in enumerate(tickers, start=1):
        group = df[df["ticker"] == ticker]
        documents = build_documents_for_ticker(ticker, group, window_size)
        if documents:
            for i in range(0, len(documents), insert_batch_size):
                batch = documents[i : i + insert_batch_size]
                result = collection.insert_many(batch)
                total_inserted += len(result.inserted_ids)
            if len(example_docs) < 2:
                example_docs.extend(documents[: 2 - len(example_docs)])

        if idx % 25 == 0 or idx == len(tickers):
            print(f"  Ticker {idx}/{len(tickers)} ({ticker}) traité — {total_inserted} documents insérés au total")

        del documents, group

    print(f"\nNombre total de documents insérés: {total_inserted}")
    print("\nExemple de documents insérés:")
    for doc in example_docs:
        print(f"\nTicker: {doc['ticker']} - Window end date: {doc['window_end_date']}")
        print(f"Label (hausse=1/baisse=0): {doc['label']}")
        print(f"Shape de la fenêtre: {len(doc['features'])}x{len(doc['features'][0])}")
        print(f"Colonnes features: {doc['metadata']['feature_columns']}")

    client.close()
    return total_inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Traitement cross-sectional staging -> MongoDB curated")
    parser.add_argument("--mysql_host", type=str, default="localhost", help="Hôte MySQL")
    parser.add_argument("--mysql_user", type=str, default="root", help="Utilisateur MySQL")
    parser.add_argument("--mysql_password", type=str, default="root", help="Mot de passe MySQL")
    parser.add_argument("--mongo_uri", type=str, default="mongodb://localhost:27017/", help="URI MongoDB")
    parser.add_argument("--window_size", type=int, default=30, help="Taille de fenêtre glissante")
    args = parser.parse_args()

    print("Récupération des données depuis MySQL (market_data_xs)...")
    df = get_mysql_data_xs(args.mysql_host, args.mysql_user, args.mysql_password, "staging")
    if df is None or df.empty:
        print("Aucune donnée récupérée depuis MySQL")
        return

    print(f"Nombre de lignes récupérées: {len(df)} ({df['ticker'].nunique()} tickers)")
    print(f"\nConstruction et insertion des séquences par ticker (fenêtre = {args.window_size} jours)...")
    total = process_all_tickers_streaming(df, args.window_size, args.mongo_uri)

    if total > 0:
        print("\nTraitement terminé avec succès!")
    else:
        print("\nErreur lors du traitement : aucun document inséré")


if __name__ == "__main__":
    main()
