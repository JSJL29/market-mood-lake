import io
import argparse
import json

import boto3
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
from numba import njit


@njit
def compute_features(close, window=20):
    """
    Calcule le rendement log et la volatilité glissante (rolling std) en
    Numba JIT, pour accélérer le calcul sur tout l'historique S&P500.

    Parameters
    ----------
    close : np.ndarray[float64]
        Prix de clôture, ordonnés chronologiquement.
    window : int
        Taille de la fenêtre glissante pour la volatilité.

    Returns
    -------
    log_return : np.ndarray[float64]
    rolling_vol : np.ndarray[float64]
        NaN sur les `window` premières observations.
    """
    n = len(close)
    log_return = np.full(n, np.nan)
    rolling_vol = np.full(n, np.nan)

    for i in range(1, n):
        log_return[i] = np.log(close[i] / close[i - 1])

    for i in range(window, n):
        rolling_vol[i] = np.std(log_return[i - window + 1:i + 1])

    return log_return, rolling_vol


@njit
def compute_rsi(close, period=14):
    """
    RSI (Relative Strength Index) de Wilder, en Numba JIT.

    Mesure si un actif a été sur-acheté (proche de 100) ou sur-vendu
    (proche de 0) sur les `period` derniers jours, en comparant
    l'ampleur moyenne des hausses et des baisses.
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        change = close[i] - close[i - 1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change

    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, n):
        if i < period:
            continue
        if i == period:
            avg_gain = np.mean(gains[1:period + 1])
            avg_loss = np.mean(losses[1:period + 1])
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit
def compute_ma_ratio(close, short_window=10, long_window=50):
    """
    Écart relatif entre moyenne mobile courte et longue :
    (MA_courte / MA_longue) - 1. Positif = tendance haussière court
    terme par rapport au long terme (signal "golden cross" continu),
    négatif = tendance baissière ("death cross"). Stationnaire (un
    ratio, pas un niveau de prix), contrairement au prix brut.
    """
    n = len(close)
    ratio = np.full(n, np.nan)

    for i in range(long_window, n):
        short_ma = np.mean(close[i - short_window + 1:i + 1])
        long_ma = np.mean(close[i - long_window + 1:i + 1])
        if long_ma != 0:
            ratio[i] = short_ma / long_ma - 1.0

    return ratio


def get_sp500_from_raw(endpoint_url, bucket_name, file_name):
    """Récupère le CSV S&P500 combiné depuis le bucket raw."""
    s3 = boto3.client('s3', endpoint_url=endpoint_url)
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    return pd.read_csv(io.BytesIO(response['Body'].read()))


def get_market_mood_from_raw(endpoint_url, bucket_name, prefix="market_mood_"):
    """
    Récupère et concatène tous les payloads JSON VIX/Fear&Greed ingérés
    depuis l'API (potentiellement plusieurs fichiers horodatés).
    """
    s3 = boto3.client('s3', endpoint_url=endpoint_url)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    vix_rows = []
    fg_rows = []
    for obj in response.get("Contents", []):
        body = s3.get_object(Bucket=bucket_name, Key=obj["Key"])
        payload = json.loads(body["Body"].read().decode("utf-8"))
        vix_rows.extend(payload.get("vix", []))
        fg_rows.extend(payload.get("fear_greed", []))

    vix_df = pd.DataFrame(vix_rows).drop_duplicates(subset="date") if vix_rows else pd.DataFrame(columns=["date"])
    fg_df = pd.DataFrame(fg_rows).drop_duplicates(subset="date") if fg_rows else pd.DataFrame(columns=["date"])
    return vix_df, fg_df


def clean_and_merge(sp500_df, vix_df, fg_df, horizon=1):
    """
    Nettoie le S&P500, calcule les features Numba, et aligne avec VIX/Fear&Greed.

    Parameters
    ----------
    horizon : int
        Nombre de jours de bourse dans le futur pour le label hausse/baisse.
        horizon=1 -> prédiction à J+1 (bruit journalier élevé).
        horizon=5 -> prédiction à J+5 (~1 semaine de bourse), signal
        potentiellement moins noyé dans le bruit court terme.
    """
    sp500_df = sp500_df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    sp500_df["date"] = pd.to_datetime(sp500_df["date"]).dt.strftime("%Y-%m-%d")

    close = sp500_df["close"].astype(np.float64).values
    log_return, rolling_vol = compute_features(close, window=20)
    sp500_df["log_return"] = log_return
    sp500_df["rolling_vol_20d"] = rolling_vol
    sp500_df["rsi_14"] = compute_rsi(close, period=14)
    sp500_df["ma_ratio"] = compute_ma_ratio(close, short_window=10, long_window=50)

    # Label : hausse (1) ou baisse (0) à J+horizon
    sp500_df["target_up"] = (sp500_df["close"].shift(-horizon) > sp500_df["close"]).astype("Int64")

    merged = sp500_df.merge(vix_df, on="date", how="left").merge(fg_df, on="date", how="left")
    merged = merged.dropna(subset=["log_return", "rolling_vol_20d", "rsi_14", "ma_ratio", "target_up"])

    return merged



def create_mysql_connection(host, user, password, database):
    try:
        return mysql.connector.connect(host=host, user=user, password=password, database=database)
    except Error as e:
        print(f"Erreur lors de la connexion à MySQL: {e}")
        return None


def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL UNIQUE,
                close DOUBLE NOT NULL,
                volume BIGINT,
                log_return DOUBLE,
                rolling_vol_20d DOUBLE,
                rsi_14 DOUBLE,
                ma_ratio DOUBLE,
                vix_close DOUBLE,
                fear_greed_score DOUBLE,
                target_up TINYINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()

        # Ajoute les nouvelles colonnes si la table existait déjà avant
        # l'introduction des indicateurs techniques (RSI, moyennes mobiles).
        for column_def in ["rsi_14 DOUBLE", "ma_ratio DOUBLE"]:
            column_name = column_def.split()[0]
            try:
                cursor.execute(f"ALTER TABLE market_data ADD COLUMN {column_def}")
                connection.commit()
                print(f"Colonne '{column_name}' ajoutée à market_data.")
            except Error:
                pass  # la colonne existe déjà
    except Error as e:
        print(f"Erreur lors de la création de la table: {e}")


def insert_data(connection, df):
    try:
        cursor = connection.cursor()
        insert_query = """
            INSERT INTO market_data (date, close, volume, log_return, rolling_vol_20d, rsi_14, ma_ratio, vix_close, fear_greed_score, target_up)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                close=VALUES(close), volume=VALUES(volume), log_return=VALUES(log_return),
                rolling_vol_20d=VALUES(rolling_vol_20d), rsi_14=VALUES(rsi_14), ma_ratio=VALUES(ma_ratio),
                vix_close=VALUES(vix_close), fear_greed_score=VALUES(fear_greed_score), target_up=VALUES(target_up)
        """
        values = [
            (
                row["date"], float(row["close"]), int(row.get("volume", 0) or 0),
                float(row["log_return"]), float(row["rolling_vol_20d"]),
                float(row["rsi_14"]), float(row["ma_ratio"]),
                float(row["vix_close"]) if pd.notna(row.get("vix_close")) else None,
                float(row["fear_greed_score"]) if pd.notna(row.get("fear_greed_score")) else None,
                int(row["target_up"]),
            )
            for _, row in df.iterrows()
        ]
        cursor.executemany(insert_query, values)
        connection.commit()
        print(f"{cursor.rowcount} lignes insérées/mises à jour avec succès.")
    except Error as e:
        print(f"Erreur lors de l'insertion des données: {e}")


def validate_data(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data")
        print(f"Nombre total de lignes: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM market_data WHERE vix_close IS NOT NULL")
        print(f"Lignes avec VIX renseigné: {cursor.fetchone()[0]}")
        cursor.execute("SELECT date, close, vix_close, target_up FROM market_data ORDER BY date DESC LIMIT 5")
        print("\nExemple des 5 dernières lignes:")
        for row in cursor.fetchall():
            print(row)
    except Error as e:
        print(f"Erreur lors de la validation des données: {e}")


def preprocess_to_staging(bucket_raw, sp500_file, db_host, db_user, db_password, endpoint_url, horizon=1):
    print("Récupération du S&P500 depuis le bucket raw...")
    sp500_df = get_sp500_from_raw(endpoint_url, bucket_raw, sp500_file)

    print("Récupération du VIX / Fear&Greed depuis le bucket raw...")
    vix_df, fg_df = get_market_mood_from_raw(endpoint_url, bucket_raw)

    print(f"Nettoyage et calcul des features (Numba), horizon de label = J+{horizon}...")
    merged = clean_and_merge(sp500_df, vix_df, fg_df, horizon=horizon)

    print("Connexion à MySQL...")
    connection = create_mysql_connection(db_host, db_user, db_password, "staging")
    if connection is None:
        return

    create_table(connection)
    insert_data(connection, merged)
    validate_data(connection)

    connection.close()
    print("\nTraitement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prépare les données marché pour le staging dans MySQL')
    parser.add_argument('--bucket_raw', type=str, default='raw', help='Nom du bucket raw')
    parser.add_argument('--sp500_file', type=str, default='sp500_combined.csv', help='Nom du fichier S&P500 dans raw')
    parser.add_argument('--db_host', type=str, required=True, help='Hôte MySQL')
    parser.add_argument('--db_user', type=str, required=True, help='Utilisateur MySQL')
    parser.add_argument('--db_password', type=str, required=True, help='Mot de passe MySQL')
    parser.add_argument('--endpoint-url', type=str, default='http://localhost:4566',
                         help='URL du endpoint S3 (LocalStack)')
    parser.add_argument('--horizon', type=int, default=1,
                         help='Horizon du label en jours de bourse (1 = J+1, 5 = J+5, etc.)')
    args = parser.parse_args()

    preprocess_to_staging(args.bucket_raw, args.sp500_file, args.db_host, args.db_user, args.db_password,
                           args.endpoint_url, args.horizon)