"""
Version cross-sectional du staging avec contexte sectoriel.

Chaque ticker garde sa propre série chronologique : les features de prix et le
label sont calculés séparément ticker par ticker. VIX/Fear & Greed restent des
variables macro partagées par date. Le contexte sectoriel ajoute une nouvelle
source de données : ETF sectoriel du ticker, rendement sectoriel et
sur/sous-performance relative du ticker face à son secteur.

Nouveaux fichiers raw optionnels attendus :
- sector_etfs.csv : produit par src.ingestion.sector_context
- ticker_sector_map.csv : produit par src.ingestion.sector_context

Si ces fichiers ne sont pas présents, le pipeline ne casse pas : les features
sectorielles sont remplies avec 0.0, mais elles n'apportent alors aucun signal.
"""

from __future__ import annotations

import argparse
import io
import itertools
import uuid
from typing import Optional

import boto3
import mysql.connector
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from mysql.connector import Error

from src.transform.preprocess_to_staging import (
    compute_features,
    compute_ma_ratio,
    compute_rsi,
    align_market_mood,
    get_market_mood_from_raw,
)


SECTOR_NUMERIC_COLUMNS = [
    "ticker_momentum_5d",
    "sector_close",
    "sector_log_return",
    "sector_rolling_vol_20d",
    "sector_momentum_5d",
    "ticker_minus_sector_log_return",
    "ticker_minus_sector_momentum_5d",
]

REQUIRED_MODEL_COLUMNS = [
    "log_return",
    "rolling_vol_20d",
    "rsi_14",
    "ma_ratio",
    "ticker_momentum_5d",
    "target_up",
    "vix_close",
    "fear_greed_score",
]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _to_float_or_none(value):
    return float(value) if pd.notna(value) else None


def _to_int_or_zero(value):
    return int(value) if pd.notna(value) else 0


def get_stocks_from_raw(endpoint_url: str, bucket_name: str, file_name: str) -> pd.DataFrame:
    """Récupère le CSV multi-actions combiné depuis le bucket raw."""
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))
    df = _normalise_columns(df)
    if "ticker" not in df.columns:
        for candidate in ["symbol", "symbols"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "ticker"})
                break
    return df


def get_optional_csv_from_raw(endpoint_url: str, bucket_name: str, file_name: str) -> Optional[pd.DataFrame]:
    """Lit un CSV raw optionnel. Retourne None si l'objet S3 n'existe pas."""
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404"}:
            print(f"Fichier raw optionnel absent : s3://{bucket_name}/{file_name}")
            return None
        raise
    df = pd.read_csv(io.BytesIO(response["Body"].read()))
    return _normalise_columns(df)


def _compute_momentum_5d(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    return np.log(close / close.shift(5))


def prepare_sector_etfs(sector_etfs_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Nettoie sector_etfs.csv et calcule les features sectorielles."""
    if sector_etfs_df is None or sector_etfs_df.empty:
        return None

    df = _normalise_columns(sector_etfs_df)
    required = {"date", "sector_etf", "sector_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sector_etfs.csv incomplet, colonnes manquantes : {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["sector_etf"] = df["sector_etf"].astype(str).str.upper().str.strip()
    df["sector_close"] = pd.to_numeric(df["sector_close"], errors="coerce")
    if "sector_volume" not in df.columns:
        df["sector_volume"] = 0
    df["sector_volume"] = pd.to_numeric(df["sector_volume"], errors="coerce").fillna(0).astype(np.int64)

    processed = []
    for sector_etf, group in df.sort_values(["sector_etf", "date"]).groupby("sector_etf"):
        group = group.copy().reset_index(drop=True)
        close = group["sector_close"].astype(np.float64).values
        sector_ret, sector_vol = compute_features(close, window=20)
        group["sector_log_return"] = sector_ret
        group["sector_rolling_vol_20d"] = sector_vol
        group["sector_momentum_5d"] = _compute_momentum_5d(group["sector_close"])
        processed.append(group)

    result = pd.concat(processed, ignore_index=True)
    return result.drop_duplicates(subset=["date", "sector_etf"])


def prepare_sector_map(sector_map_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if sector_map_df is None or sector_map_df.empty:
        return None

    df = _normalise_columns(sector_map_df)
    if "ticker" not in df.columns:
        raise ValueError("ticker_sector_map.csv doit contenir une colonne ticker.")
    if "sector" not in df.columns:
        df["sector"] = "UNKNOWN"
    if "sector_etf" not in df.columns:
        df["sector_etf"] = "SPY"

    df = df[["ticker", "sector", "sector_etf"]].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["sector"] = df["sector"].fillna("UNKNOWN").astype(str)
    df["sector_etf"] = df["sector_etf"].fillna("SPY").astype(str).str.upper().str.strip()
    return df.drop_duplicates(subset=["ticker"])


def add_neutral_sector_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des colonnes sectorielles neutres si aucune source n'est fournie."""
    df = df.copy()
    df["sector"] = "UNKNOWN"
    df["sector_etf"] = "SPY"
    df["sector_close"] = 0.0
    df["sector_log_return"] = 0.0
    df["sector_rolling_vol_20d"] = 0.0
    df["sector_momentum_5d"] = 0.0
    df["ticker_minus_sector_log_return"] = df["log_return"].astype(float)
    df["ticker_minus_sector_momentum_5d"] = df["ticker_momentum_5d"].astype(float)
    return df


def merge_sector_context(
    stocks_df: pd.DataFrame,
    sector_etfs_df: Optional[pd.DataFrame],
    sector_map_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    sector_etfs = prepare_sector_etfs(sector_etfs_df)
    sector_map = prepare_sector_map(sector_map_df)

    if sector_etfs is None or sector_map is None:
        print("Contexte sectoriel non disponible : colonnes sectorielles neutres ajoutées.")
        return add_neutral_sector_columns(stocks_df)

    merged = stocks_df.merge(sector_map, on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("UNKNOWN")
    merged["sector_etf"] = merged["sector_etf"].fillna("SPY").astype(str).str.upper().str.strip()

    merged = merged.merge(sector_etfs, on=["date", "sector_etf"], how="left")

    for col in ["sector_close", "sector_log_return", "sector_rolling_vol_20d", "sector_momentum_5d"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["ticker_minus_sector_log_return"] = merged["log_return"] - merged["sector_log_return"]
    merged["ticker_minus_sector_momentum_5d"] = merged["ticker_momentum_5d"] - merged["sector_momentum_5d"]
    return merged


def clean_and_merge_xs(
    stocks_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    fg_df: pd.DataFrame,
    sector_etfs_df: Optional[pd.DataFrame] = None,
    sector_map_df: Optional[pd.DataFrame] = None,
    horizon: int = 1,
    max_tickers: Optional[int] = None,
) -> pd.DataFrame:
    """Nettoie les données multi-actions et fusionne macro + secteur."""
    if horizon < 1:
        raise ValueError("horizon doit être supérieur ou égal à 1")
    stocks_df = _normalise_columns(stocks_df)
    if "ticker" not in stocks_df.columns:
        for candidate in ["symbol", "symbols"]:
            if candidate in stocks_df.columns:
                stocks_df = stocks_df.rename(columns={candidate: "ticker"})
                break
    if "ticker" not in stocks_df.columns:
        raise ValueError("Le fichier actions doit contenir ticker/symbol.")

    stocks_df = stocks_df.dropna(subset=["close", "ticker"]).copy()
    stocks_df["ticker"] = stocks_df["ticker"].astype(str).str.upper().str.strip()
    stocks_df["date"] = pd.to_datetime(stocks_df["date"]).dt.strftime("%Y-%m-%d")
    stocks_df["close"] = pd.to_numeric(stocks_df["close"], errors="coerce")
    if "volume" not in stocks_df.columns:
        stocks_df["volume"] = 0
    stocks_df["volume"] = pd.to_numeric(stocks_df["volume"], errors="coerce").fillna(0).astype(np.int64)
    if max_tickers is not None:
        if max_tickers < 1:
            raise ValueError("max_tickers doit être supérieur ou égal à 1")
        kept = stocks_df.groupby("ticker").size().nlargest(max_tickers).index
        stocks_df = stocks_df[stocks_df["ticker"].isin(kept)].copy()
        print(f"Mode borné: {len(kept)} tickers avec le plus d'historique conservés.")

    processed_groups = []
    tickers = stocks_df["ticker"].unique()
    print(f"Calcul des features pour {len(tickers)} tickers...")

    for ticker in tickers:
        group = stocks_df[stocks_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(group) < 90:
            continue

        close = group["close"].astype(np.float64).values
        log_return, rolling_vol = compute_features(close, window=20)
        group["log_return"] = log_return
        group["rolling_vol_20d"] = rolling_vol
        group["rsi_14"] = compute_rsi(close, period=14)
        group["ma_ratio"] = compute_ma_ratio(close, short_window=10, long_window=50)
        group["ticker_momentum_5d"] = _compute_momentum_5d(group["close"])

        # Label calculé DANS la série du ticker, jamais à cheval sur deux tickers.
        future_close = group["close"].shift(-horizon)
        group["target_up"] = (future_close > group["close"]).astype("Int64")
        group.loc[future_close.isna(), "target_up"] = pd.NA
        processed_groups.append(group)

    if not processed_groups:
        raise RuntimeError("Aucun ticker n'a assez d'historique pour le staging cross-sectional.")

    all_stocks = pd.concat(processed_groups, ignore_index=True)
    all_stocks = merge_sector_context(all_stocks, sector_etfs_df, sector_map_df)

    merged = align_market_mood(all_stocks, vix_df, fg_df)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    for col in SECTOR_NUMERIC_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged = merged.dropna(subset=REQUIRED_MODEL_COLUMNS)
    return merged


def create_mysql_connection(host: str, user: str, password: str, database: str):
    try:
        return mysql.connector.connect(host=host, user=user, password=password, database=database)
    except Error as e:
        raise RuntimeError(f"Erreur lors de la connexion à MySQL: {e}") from e


def create_table_xs(connection) -> None:
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
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
                ticker_momentum_5d DOUBLE,
                vix_close DOUBLE,
                fear_greed_score DOUBLE,
                sector VARCHAR(64),
                sector_etf VARCHAR(16),
                sector_close DOUBLE,
                sector_log_return DOUBLE,
                sector_rolling_vol_20d DOUBLE,
                sector_momentum_5d DOUBLE,
                ticker_minus_sector_log_return DOUBLE,
                ticker_minus_sector_momentum_5d DOUBLE,
                target_up TINYINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_date_ticker (date, ticker)
            )
            """
        )
        connection.commit()

        column_defs = [
            "rsi_14 DOUBLE",
            "ma_ratio DOUBLE",
            "ticker_momentum_5d DOUBLE",
            "sector VARCHAR(64)",
            "sector_etf VARCHAR(16)",
            "sector_close DOUBLE",
            "sector_log_return DOUBLE",
            "sector_rolling_vol_20d DOUBLE",
            "sector_momentum_5d DOUBLE",
            "ticker_minus_sector_log_return DOUBLE",
            "ticker_minus_sector_momentum_5d DOUBLE",
        ]
        for column_def in column_defs:
            column_name = column_def.split()[0]
            try:
                cursor.execute(f"ALTER TABLE market_data_xs ADD COLUMN {column_def}")
                connection.commit()
                print(f"Colonne '{column_name}' ajoutée à market_data_xs.")
            except Error as exc:
                if exc.errno != 1060:  # duplicate column
                    raise
    except Error as e:
        raise RuntimeError(f"Erreur lors de la création de la table: {e}") from e


def insert_data_xs(connection, df: pd.DataFrame, batch_size: int = 5000, table_name="market_data_xs") -> None:
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Nom de table invalide")
    cursor = connection.cursor()
    try:
        insert_query = """
            INSERT INTO {table_name} (
                date, ticker, close, volume,
                log_return, rolling_vol_20d, rsi_14, ma_ratio, ticker_momentum_5d,
                vix_close, fear_greed_score,
                sector, sector_etf, sector_close, sector_log_return, sector_rolling_vol_20d,
                sector_momentum_5d, ticker_minus_sector_log_return, ticker_minus_sector_momentum_5d,
                target_up
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                close=VALUES(close),
                volume=VALUES(volume),
                log_return=VALUES(log_return),
                rolling_vol_20d=VALUES(rolling_vol_20d),
                rsi_14=VALUES(rsi_14),
                ma_ratio=VALUES(ma_ratio),
                ticker_momentum_5d=VALUES(ticker_momentum_5d),
                vix_close=VALUES(vix_close),
                fear_greed_score=VALUES(fear_greed_score),
                sector=VALUES(sector),
                sector_etf=VALUES(sector_etf),
                sector_close=VALUES(sector_close),
                sector_log_return=VALUES(sector_log_return),
                sector_rolling_vol_20d=VALUES(sector_rolling_vol_20d),
                sector_momentum_5d=VALUES(sector_momentum_5d),
                ticker_minus_sector_log_return=VALUES(ticker_minus_sector_log_return),
                ticker_minus_sector_momentum_5d=VALUES(ticker_minus_sector_momentum_5d),
                target_up=VALUES(target_up)
        """.format(table_name=table_name)

        columns = [
            "date", "ticker", "close", "volume", "log_return", "rolling_vol_20d", "rsi_14",
            "ma_ratio", "ticker_momentum_5d", "vix_close", "fear_greed_score", "sector", "sector_etf",
            "sector_close", "sector_log_return", "sector_rolling_vol_20d", "sector_momentum_5d",
            "ticker_minus_sector_log_return", "ticker_minus_sector_momentum_5d", "target_up",
        ]
        rows = df[columns].itertuples(index=False, name=None)
        total_inserted = 0
        while True:
            raw_batch = list(itertools.islice(rows, batch_size))
            if not raw_batch:
                break
            batch = [
                (
                    row[0], row[1], float(row[2]), _to_int_or_zero(row[3]),
                    *[_to_float_or_none(value) for value in row[4:11]],
                    str(row[11]), str(row[12]),
                    *[_to_float_or_none(value) for value in row[13:19]], int(row[19]),
                )
                for row in raw_batch
            ]
            cursor.executemany(insert_query, batch)
            connection.commit()
            total_inserted += len(batch)
            print(f"  {total_inserted}/{len(df)} lignes insérées...")
        print(f"{total_inserted} lignes insérées/mises à jour avec succès.")
    except Exception as e:
        connection.rollback()
        raise RuntimeError(f"Erreur lors de l'insertion des données: {e}") from e
    finally:
        cursor.close()


def validate_data_xs(connection, table_name="market_data_xs") -> None:
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Nom de table invalide")
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        print(f"Nombre total de lignes: {cursor.fetchone()[0]}")
        cursor.execute(f"SELECT COUNT(DISTINCT ticker) FROM {table_name}")
        print(f"Nombre de tickers: {cursor.fetchone()[0]}")
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE vix_close IS NOT NULL")
        print(f"Lignes avec VIX renseigné: {cursor.fetchone()[0]}")
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE sector_etf IS NOT NULL")
        print(f"Lignes avec secteur/ETF renseigné: {cursor.fetchone()[0]}")
        cursor.execute(
            """
            SELECT ticker, sector, sector_etf, date, log_return, sector_log_return,
                   ticker_minus_sector_log_return
            FROM {table_name}
            ORDER BY date DESC, ticker ASC
            LIMIT 5
            """.format(table_name=table_name)
        )
        print("\nExemple des 5 dernières lignes enrichies secteur:")
        for row in cursor.fetchall():
            print(row)
    except Error as e:
        raise RuntimeError(f"Erreur lors de la validation des données: {e}") from e
    finally:
        if 'cursor' in locals():
            cursor.close()


def preprocess_to_staging_xs(
    bucket_raw: str,
    stocks_file: str,
    db_host: str,
    db_user: str,
    db_password: str,
    endpoint_url: str,
    horizon: int = 1,
    sector_etfs_file: str = "sector_etfs.csv",
    sector_map_file: str = "ticker_sector_map.csv",
    max_tickers: Optional[int] = None,
) -> None:
    print("Récupération des actions depuis le bucket raw...")
    stocks_df = get_stocks_from_raw(endpoint_url, bucket_raw, stocks_file)

    print("Récupération du VIX / Fear&Greed depuis le bucket raw...")
    vix_df, fg_df = get_market_mood_from_raw(endpoint_url, bucket_raw)

    print("Récupération du contexte sectoriel depuis le bucket raw...")
    sector_etfs_df = get_optional_csv_from_raw(endpoint_url, bucket_raw, sector_etfs_file)
    sector_map_df = get_optional_csv_from_raw(endpoint_url, bucket_raw, sector_map_file)

    print(f"Nettoyage et calcul des features par ticker, horizon de label = J+{horizon}...")
    merged = clean_and_merge_xs(
        stocks_df,
        vix_df,
        fg_df,
        sector_etfs_df=sector_etfs_df,
        sector_map_df=sector_map_df,
        horizon=horizon,
        max_tickers=max_tickers,
    )

    print("Connexion à MySQL...")
    connection = create_mysql_connection(db_host, db_user, db_password, "staging")
    temporary_table = f"market_data_xs_load_{uuid.uuid4().hex}"
    backup_table = f"market_data_xs_backup_{uuid.uuid4().hex}"
    try:
        create_table_xs(connection)
        cursor = connection.cursor()
        cursor.execute(f"CREATE TABLE {temporary_table} LIKE market_data_xs")
        connection.commit()
        cursor.close()
        insert_data_xs(connection, merged, table_name=temporary_table)
        validate_data_xs(connection, table_name=temporary_table)
        cursor = connection.cursor()
        cursor.execute(
            f"RENAME TABLE market_data_xs TO {backup_table}, {temporary_table} TO market_data_xs"
        )
        cursor.execute(f"DROP TABLE {backup_table}")
        connection.commit()
        cursor.close()
    finally:
        try:
            cursor = connection.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {temporary_table}")
            connection.commit()
            cursor.close()
        finally:
            connection.close()
    print("\nTraitement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prépare les données multi-actions cross-sectional pour MySQL")
    parser.add_argument("--bucket_raw", type=str, default="raw", help="Nom du bucket raw")
    parser.add_argument("--stocks_file", type=str, default="stocks_combined.csv", help="Nom du fichier multi-actions dans raw")
    parser.add_argument("--db_host", type=str, required=True, help="Hôte MySQL")
    parser.add_argument("--db_user", type=str, required=True, help="Utilisateur MySQL")
    parser.add_argument("--db_password", type=str, required=True, help="Mot de passe MySQL")
    parser.add_argument("--endpoint-url", type=str, default="http://localhost:4566", help="URL endpoint S3 LocalStack")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon du label en jours de bourse")
    parser.add_argument("--sector_etfs_file", type=str, default="sector_etfs.csv")
    parser.add_argument("--sector_map_file", type=str, default="ticker_sector_map.csv")
    parser.add_argument("--max_tickers", type=int, default=None)
    args = parser.parse_args()

    preprocess_to_staging_xs(
        args.bucket_raw,
        args.stocks_file,
        args.db_host,
        args.db_user,
        args.db_password,
        args.endpoint_url,
        args.horizon,
        args.sector_etfs_file,
        args.sector_map_file,
        args.max_tickers,
    )
