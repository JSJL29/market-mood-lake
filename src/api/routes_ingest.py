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
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from numba import njit
import mysql.connector

router = APIRouter()

MYSQL_CONFIG = {
    'host': 'mysql',
    'user': 'root',
    'password': 'root',
    'database': 'staging'
}


class MarketTick(BaseModel):
    date: str
    close: float
    volume: int = 0
    vix_close: float | None = None


class IngestPayload(BaseModel):
    data: List[MarketTick]


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


def _insert_naive(ticks: List[MarketTick], log_returns, vix_z):
    """Insertion ligne par ligne (naïve)."""
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    for i, t in enumerate(ticks):
        cursor.execute(
            """INSERT INTO market_data (date, close, volume, log_return, vix_close)
               VALUES (%s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE close=VALUES(close), log_return=VALUES(log_return)""",
            (t.date, t.close, t.volume, float(log_returns[i]), t.vix_close),
        )
    conn.commit()
    cursor.close()
    conn.close()


def _insert_batch(ticks: List[MarketTick], log_returns, vix_z):
    """Insertion par batch avec executemany (optimisée)."""
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    values = [
        (t.date, t.close, t.volume, float(log_returns[i]), t.vix_close)
        for i, t in enumerate(ticks)
    ]
    cursor.executemany(
        """INSERT INTO market_data (date, close, volume, log_return, vix_close)
           VALUES (%s, %s, %s, %s, %s)
           ON DUPLICATE KEY UPDATE close=VALUES(close), log_return=VALUES(log_return)""",
        values,
    )
    conn.commit()
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
    _insert_naive(payload.data, log_returns, vix_z)

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

    close_arr = np.array([t.close for t in payload.data], dtype=np.float64)
    vix_arr = np.array([t.vix_close if t.vix_close is not None else 0.0 for t in payload.data], dtype=np.float64)

    log_returns, vix_z = _vectorized_features(close_arr, vix_arr)
    _insert_batch(payload.data, log_returns, vix_z)

    elapsed = time.perf_counter() - start
    return {
        "endpoint": "ingest_fast",
        "batch_size": len(payload.data),
        "elapsed_seconds": round(elapsed, 6),
    }
