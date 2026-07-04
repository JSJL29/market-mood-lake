"""
Version cross-sectional du staging : au lieu d'un seul indice (S&P500),
traite plusieurs actions individuelles. Chaque ticker a sa propre série
chronologique de prix ; les features (rendement log, volatilité) et le
label (hausse/baisse à horizon J+N) sont calculés SÉPARÉMENT pour
chaque ticker, jamais en mélangeant les séries de deux actions
différentes. Le VIX et le Fear & Greed Index, eux, sont partagés par
tous les tickers pour une même date (ce sont des indicateurs de marché
global, pas propres à une action).

L'intérêt de cette approche (par rapport au staging mono-indice) est
documenté dans Fischer & Krauss (2018) : entraîner un même modèle sur
de nombreuses actions ("coupe transversale") multiplie le nombre
d'échantillons d'entraînement réellement indépendants, par rapport à un
seul indice observé sur la même période.
"""
import io
import argparse

import boto3
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error

from src.transform.preprocess_to_staging import compute_features, compute_rsi, compute_ma_ratio, get_market_mood_from_raw


def get_stocks_from_raw(endpoint_url, bucket_name, file_name):
    """Récupère le CSV multi-actions combiné depuis le bucket raw."""
    s3 = boto3.client('s3', endpoint_url=endpoint_url)
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    return pd.read_csv(io.BytesIO(response['Body'].read()))


def clean_and_merge_xs(stocks_df, vix_df, fg_df, horizon=1):
    """
    Nettoie les données multi-actions, calcule les features par ticker,
    et fusionne avec VIX/Fear&Greed (partagés entre tickers, sur la date).
    """
    stocks_df = stocks_df.dropna(subset=["close", "ticker"]).copy()
    stocks_df["date"] = pd.to_datetime(stocks_df["date"]).dt.strftime("%Y-%m-%d")

    processed_groups = []
    tickers = stocks_df["ticker"].unique()
    print(f"Calcul des features pour {len(tickers)} tickers...")

    for ticker in tickers:
        group = stocks_df[stocks_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(group) < 90:  # historique insuffisant (warm-up ma_ratio 50j + fenêtre 30j + marge)
            continue

        close = group["close"].astype(np.float64).values
        log_return, rolling_vol = compute_features(close, window=20)
        group["log_return"] = log_return
        group["rolling_vol_20d"] = rolling_vol
        group["rsi_14"] = compute_rsi(close, period=14)
        group["ma_ratio"] = compute_ma_ratio(close, short_window=10, long_window=50)

        # Label calculé DANS la série du ticker, jamais à cheval sur deux tickers
        group["target_up"] = (group["close"].shift(-horizon) > group["close"]).astype("Int64")

        processed_groups.append(group)

    all_stocks = pd.concat(processed_groups, ignore_index=True)

    merged = all_stocks.merge(vix_df, on="date", how="left").merge(fg_df, on="date", how="left")
    merged = merged.dropna(subset=["log_return", "rolling_vol_20d", "rsi_14", "ma_ratio", "target_up"])

    return merged


def create_mysql_connection(host, user, password, database):
    try:
        return mysql.connector.connect(host=host, user=user, password=password, database=database)
    except Error as e:
        print(f"Erreur lors de la connexion à MySQL: {e}")
        return None


def create_table_xs(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_xs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                ticker VARCHAR(16) NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT,
                log_return DOUBLE,
                rolling_vol_20d DOUBLE,
                rsi_14 DOUBLE,
                ma_ratio DOUBLE,
                vix_close DOUBLE,
                fear_greed_score DOUBLE,
                target_up TINYINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_date_ticker (date, ticker)
            )
        """)
        connection.commit()

        for column_def in ["rsi_14 DOUBLE", "ma_ratio DOUBLE"]:
            column_name = column_def.split()[0]
            try:
                cursor.execute(f"ALTER TABLE market_data_xs ADD COLUMN {column_def}")
                connection.commit()
                print(f"Colonne '{column_name}' ajoutée à market_data_xs.")
            except Error:
                pass  # la colonne existe déjà
    except Error as e:
        print(f"Erreur lors de la création de la table: {e}")


def insert_data_xs(connection, df, batch_size=5000):
    try:
        cursor = connection.cursor()
        insert_query = """
            INSERT INTO market_data_xs (date, ticker, close, volume, log_return, rolling_vol_20d, rsi_14, ma_ratio, vix_close, fear_greed_score, target_up)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                close=VALUES(close), volume=VALUES(volume), log_return=VALUES(log_return),
                rolling_vol_20d=VALUES(rolling_vol_20d), rsi_14=VALUES(rsi_14), ma_ratio=VALUES(ma_ratio),
                vix_close=VALUES(vix_close), fear_greed_score=VALUES(fear_greed_score), target_up=VALUES(target_up)
        """
        values = [
            (
                row["date"], row["ticker"], float(row["close"]), int(row.get("volume", 0) or 0),
                float(row["log_return"]), float(row["rolling_vol_20d"]),
                float(row["rsi_14"]), float(row["ma_ratio"]),
                float(row["vix_close"]) if pd.notna(row.get("vix_close")) else None,
                float(row["fear_greed_score"]) if pd.notna(row.get("fear_greed_score")) else None,
                int(row["target_up"]),
            )
            for _, row in df.iterrows()
        ]

        total_inserted = 0
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            connection.commit()
            total_inserted += len(batch)
            print(f"  {total_inserted}/{len(values)} lignes insérées...")

        print(f"{total_inserted} lignes insérées/mises à jour avec succès.")
    except Error as e:
        print(f"Erreur lors de l'insertion des données: {e}")


def validate_data_xs(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data_xs")
        print(f"Nombre total de lignes: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM market_data_xs")
        print(f"Nombre de tickers: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM market_data_xs WHERE vix_close IS NOT NULL")
        print(f"Lignes avec VIX renseigné: {cursor.fetchone()[0]}")
    except Error as e:
        print(f"Erreur lors de la validation des données: {e}")


def preprocess_to_staging_xs(bucket_raw, stocks_file, db_host, db_user, db_password, endpoint_url, horizon=1):
    print("Récupération des actions depuis le bucket raw...")
    stocks_df = get_stocks_from_raw(endpoint_url, bucket_raw, stocks_file)

    print("Récupération du VIX / Fear&Greed depuis le bucket raw...")
    vix_df, fg_df = get_market_mood_from_raw(endpoint_url, bucket_raw)

    print(f"Nettoyage et calcul des features par ticker (Numba), horizon de label = J+{horizon}...")
    merged = clean_and_merge_xs(stocks_df, vix_df, fg_df, horizon=horizon)

    print("Connexion à MySQL...")
    connection = create_mysql_connection(db_host, db_user, db_password, "staging")
    if connection is None:
        return

    create_table_xs(connection)
    insert_data_xs(connection, merged)
    validate_data_xs(connection)

    connection.close()
    print("\nTraitement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prépare les données multi-actions (cross-sectional) pour MySQL')
    parser.add_argument('--bucket_raw', type=str, default='raw', help='Nom du bucket raw')
    parser.add_argument('--stocks_file', type=str, default='stocks_combined.csv', help='Nom du fichier multi-actions dans raw')
    parser.add_argument('--db_host', type=str, required=True, help='Hôte MySQL')
    parser.add_argument('--db_user', type=str, required=True, help='Utilisateur MySQL')
    parser.add_argument('--db_password', type=str, required=True, help='Mot de passe MySQL')
    parser.add_argument('--endpoint-url', type=str, default='http://localhost:4566',
                         help='URL du endpoint S3 (LocalStack)')
    parser.add_argument('--horizon', type=int, default=1,
                         help='Horizon du label en jours de bourse (1 = J+1, 5 = J+5, etc.)')
    args = parser.parse_args()

    preprocess_to_staging_xs(args.bucket_raw, args.stocks_file, args.db_host, args.db_user, args.db_password,
                              args.endpoint_url, args.horizon)