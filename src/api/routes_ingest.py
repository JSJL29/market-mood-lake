"""
Endpoints avancés du niveau "Data Lakes - Projet Final" :
/ingest (naïf) vs /ingest_fast (optimisé), avec mesure de performance
sur batch de 1 et batch de 100 éléments.

Optimisations utilisées dans /ingest_fast par rapport à /ingest :
- Vectorisation NumPy du calcul des features (rendement log, z-score VIX)
  au lieu d'une boucle Python élément par élément.
- Compilation JIT avec Numba (@njit) sur la fonction de calcul vectorisé.
- Insertion MySQL par batch (executemany) au lieu d'un insert par ligne.
"""
import time
import os
from datetime import date
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from numba import njit
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool

router = APIRouter()

MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'mysql'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', 'root'),
    'database': os.getenv('MYSQL_DATABASE', 'staging'),
}


class MarketTick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date
    close: float = Field(gt=0)
    volume: int = Field(default=0, ge=0)
    vix_close: float | None = Field(default=None, ge=0)


class IngestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: List[MarketTick] = Field(min_length=1, max_length=10_000)
    benchmark: bool = False


_fast_pool = None
_ensured_tables = set()


def _fast_connection():
    """Reuse established DB connections; this matters most for micro-batches."""
    global _fast_pool
    if _fast_pool is None:
        _fast_pool = MySQLConnectionPool(
            pool_name="ingest_fast", pool_size=5, pool_reset_session=False, **MYSQL_CONFIG
        )
    return _fast_pool.get_connection()


@njit
def _vectorized_features(close_arr, vix_arr):
    """Calcul vectorisé + JIT du rendement log et du z-score du VIX sur le batch."""
    n = len(close_arr)
    log_returns = np.zeros(n)
    for i in range(1, n):
        log_returns[i] = np.log(close_arr[i] / close_arr[i - 1])

    vix_mean = np.mean(vix_arr)
    vix_std = np.std(vix_arr) if np.std(vix_arr) > 0 else 1.0
    vix_z = (vix_arr - vix_mean) / vix_std

    return log_returns, vix_z


def _naive_features(ticks: List[MarketTick]):
    """Version naïve : boucle Python pure, élément par élément."""
    log_returns = []
    closes = [t.close for t in ticks]
    for i in range(len(ticks)):
        if i == 0:
            log_returns.append(0.0)
        else:
            log_returns.append(float(np.log(closes[i] / closes[i - 1])))

    vix_values = [t.vix_close if t.vix_close is not None else 0.0 for t in ticks]
    mean_vix = sum(vix_values) / len(vix_values) if vix_values else 0.0
    variance = sum((v - mean_vix) ** 2 for v in vix_values) / len(vix_values) if vix_values else 0.0
    std_vix = variance ** 0.5 if variance > 0 else 1.0
    vix_z = [(v - mean_vix) / std_vix for v in vix_values]

    return log_returns, vix_z


def _table_for(benchmark: bool) -> str:
    return "market_data_ingest_benchmark" if benchmark else "market_data_ingest"


def _ensure_ingest_table(cursor, table: str) -> None:
    if table in _ensured_tables:
        return
    cursor.execute(
        f"""CREATE TABLE IF NOT EXISTS {table} (
               date DATE PRIMARY KEY, close DOUBLE NOT NULL, volume BIGINT,
               log_return DOUBLE, vix_close DOUBLE, vix_z DOUBLE
           )"""
    )
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN vix_z DOUBLE")
    except mysql.connector.Error as exc:
        if exc.errno != 1060:
            raise
    _ensured_tables.add(table)


def _insert_naive(ticks: List[MarketTick], log_returns, vix_z, benchmark: bool = False):
    """Insertion ligne par ligne (naïve)."""
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    try:
        table = _table_for(benchmark)
        _ensure_ingest_table(cursor, table)
        for i, t in enumerate(ticks):
            cursor.execute(
                f"""INSERT INTO {table} (date, close, volume, log_return, vix_close, vix_z)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE close=VALUES(close), volume=VALUES(volume),
                   log_return=VALUES(log_return), vix_close=VALUES(vix_close), vix_z=VALUES(vix_z)""",
                (t.date, t.close, t.volume, float(log_returns[i]), t.vix_close, float(vix_z[i])),
            )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _insert_batch(ticks: List[MarketTick], log_returns, vix_z, benchmark: bool = False):
    """Insertion par batch avec executemany (optimisée)."""
    conn = _fast_connection()
    cursor = conn.cursor()
    try:
        table = _table_for(benchmark)
        _ensure_ingest_table(cursor, table)
        values = [
            (t.date, t.close, t.volume, float(log_returns[i]), t.vix_close, float(vix_z[i]))
            for i, t in enumerate(ticks)
        ]
        cursor.executemany(
            f"""INSERT INTO {table} (date, close, volume, log_return, vix_close, vix_z)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE close=VALUES(close), volume=VALUES(volume),
               log_return=VALUES(log_return), vix_close=VALUES(vix_close), vix_z=VALUES(vix_z)""",
            values,
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


@router.post("/ingest")
async def ingest(payload: IngestPayload):
    """
    Endpoint d'ingestion naïf : boucle Python pour les features,
    insertion MySQL ligne par ligne.
    """
    start = time.perf_counter()

    log_returns, vix_z = _naive_features(payload.data)
    _insert_naive(payload.data, log_returns, vix_z, benchmark=payload.benchmark)

    elapsed = time.perf_counter() - start
    return {
        "endpoint": "ingest",
        "batch_size": len(payload.data),
        "elapsed_seconds": round(elapsed, 6),
    }


@router.post("/ingest_fast")
async def ingest_fast(payload: IngestPayload):
    """
    Endpoint d'ingestion optimisé : NumPy vectorisé + Numba JIT pour les
    features, insertion MySQL par batch (executemany).
    """
    start = time.perf_counter()

    if not payload.data:
        return {"endpoint": "ingest_fast", "batch_size": 0, "elapsed_seconds": 0.0}

    close_arr = np.array([t.close for t in payload.data], dtype=np.float64)
    vix_arr = np.array([t.vix_close if t.vix_close is not None else 0.0 for t in payload.data], dtype=np.float64)

    # Calling Numba for one row costs more than the calculation itself.
    if len(payload.data) == 1:
        log_returns = np.zeros(1, dtype=np.float64)
        vix_z = np.zeros(1, dtype=np.float64)
    else:
        log_returns, vix_z = _vectorized_features(close_arr, vix_arr)
    _insert_batch(payload.data, log_returns, vix_z, benchmark=payload.benchmark)

    elapsed = time.perf_counter() - start
    return {
        "endpoint": "ingest_fast",
        "batch_size": len(payload.data),
        "elapsed_seconds": round(elapsed, 6),
    }


@router.delete("/ingest/benchmark-data")
async def reset_benchmark_data():
    """Reset only the isolated benchmark table, never pipeline data."""
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    try:
        table = _table_for(True)
        _ensure_ingest_table(cursor, table)
        cursor.execute(f"DELETE FROM {table}")
        deleted = cursor.rowcount
        conn.commit()
        return {"table": table, "deleted_rows": deleted}
    finally:
        cursor.close()
        conn.close()
