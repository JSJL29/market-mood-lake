import os
from datetime import date, datetime
from typing import List, Optional

import boto3
import mysql.connector
from fastapi import FastAPI, HTTPException, Query
from botocore.config import Config
from pymongo import MongoClient

from src.api.routes_ingest import router as ingest_router

app = FastAPI(title="Market Mood Lake API")
app.include_router(ingest_router)


class DatabaseConnections:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localstack:4566"),
            config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 1}),
        )
        self.mysql_config = {
            "host": os.getenv("MYSQL_HOST", "mysql"),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", "root"),
            "database": os.getenv("MYSQL_DATABASE", "staging"),
            "connection_timeout": 5,
        }
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
        self.mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
        self.mongo_db = self.mongo_client[os.getenv("MONGO_DB", "curated")]


db = DatabaseConnections()


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _mysql_scalar(query: str, params=None):
    conn = mysql.connector.connect(**db.mysql_config)
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(query, params or [])
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            conn.close()


def _s3_objects(bucket: str, max_items: int | None = None):
    objects = []
    paginator = db.s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects.extend(page.get("Contents", []))
        if max_items is not None and len(objects) >= max_items:
            return objects[:max_items]
    return objects


@app.get("/raw/", response_model=List[dict])
def get_raw_data(
    bucket: Optional[str] = Query("raw", description="Nom du bucket S3"),
    limit: Optional[int] = Query(10, ge=1, le=1000, description="Nombre maximum d'objets à lister"),
):
    """Liste les objets disponibles dans la zone raw S3."""
    try:
        contents = _s3_objects(bucket, limit)
        return [
            {
                "key": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            }
            for obj in contents
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture depuis S3: {exc}")


@app.get("/staging/", response_model=List[dict])
def get_staging_data(
    start_date: Optional[date] = Query(None, description="Date de début YYYY-MM-DD"),
    end_date: Optional[date] = Query(None, description="Date de fin YYYY-MM-DD"),
    limit: Optional[int] = Query(100, ge=1, le=5000, description="Nombre maximum de lignes"),
    table: Optional[str] = Query("market_data", description="market_data ou market_data_xs"),
):
    """Récupère les données staging depuis MySQL."""
    if table not in {"market_data", "market_data_xs"}:
        raise HTTPException(status_code=400, detail="table doit valoir market_data ou market_data_xs")

    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**db.mysql_config)
        cursor = conn.cursor(dictionary=True)
        query = f"SELECT * FROM {table} WHERE 1=1"
        params = []
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        query += " ORDER BY date DESC LIMIT %s"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            for key, value in list(row.items()):
                row[key] = _iso(value)
        return rows
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture depuis MySQL: {exc}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


@app.get("/curated/", response_model=List[dict])
def get_curated_data(
    limit: Optional[int] = Query(10, ge=1, le=1000, description="Nombre maximum de documents à retourner"),
    label: Optional[int] = Query(None, ge=0, le=1, description="Filtrer par label 1=hausse, 0=baisse"),
    collection: Optional[str] = Query("market_sequences", description="Collection MongoDB curated"),
):
    """Récupère des documents curated depuis MongoDB."""
    allowed = {"market_sequences", "market_sequences_xs", "model_runs", "benchmark_runs"}
    if collection not in allowed:
        raise HTTPException(status_code=400, detail=f"collection doit être dans {sorted(allowed)}")
    try:
        query = {}
        if label is not None and collection.startswith("market_sequences"):
            query["label"] = label
        sort_field = "window_end_date" if collection.startswith("market_sequences") else "created_at"
        docs = list(
            db.mongo_db[collection]
            .find(query, {"_id": 0})
            .sort(sort_field, -1)
            .limit(limit)
        )
        return docs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture depuis MongoDB: {exc}")

@app.get("/health")
def health_check():
    """Vérifie l'état de l'API et des connexions aux services."""
    status = {
        "api_status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": {},
    }
    try:
        db.s3_client.list_buckets()
        status["connections"]["s3"] = True
    except Exception as exc:
        status["connections"]["s3"] = False
        status["connections"]["s3_error"] = str(exc)

    try:
        conn = mysql.connector.connect(**db.mysql_config)
        conn.close()
        status["connections"]["mysql"] = True
    except Exception as exc:
        status["connections"]["mysql"] = False
        status["connections"]["mysql_error"] = str(exc)

    try:
        db.mongo_client.server_info()
        status["connections"]["mongodb"] = True
    except Exception as exc:
        status["connections"]["mongodb"] = False
        status["connections"]["mongodb_error"] = str(exc)

    if not all(status["connections"].get(name, False) for name in ("s3", "mysql", "mongodb")):
        status["api_status"] = "degraded"
    return status


@app.get("/stats")
def stats():
    """Métriques de remplissage raw/staging/curated + derniers runs ML/benchmarks."""
    result = {
        "timestamp": datetime.now().isoformat(),
        "raw": {},
        "staging": {},
        "curated": {},
        "ml": {},
        "benchmarks": {},
        "errors": {},
    }

    try:
        objects = _s3_objects("raw")
        latest = max(objects, key=lambda obj: obj["LastModified"]) if objects else None
        result["raw"] = {
            "object_count": len(objects),
            "total_size_bytes": int(sum(obj.get("Size", 0) for obj in objects)),
            "latest_object": None
            if latest is None
            else {
                "key": latest["Key"],
                "size_bytes": latest["Size"],
                "last_modified": latest["LastModified"].isoformat(),
            },
        }
    except Exception as exc:
        result["raw"] = {"error": str(exc)}

    for table in ["market_data", "market_data_xs"]:
        try:
            row_count = _mysql_scalar(f"SELECT COUNT(*) FROM {table}")
            min_date = _mysql_scalar(f"SELECT MIN(date) FROM {table}")
            max_date = _mysql_scalar(f"SELECT MAX(date) FROM {table}")
            result["staging"][table] = {
                "row_count": int(row_count or 0),
                "min_date": _iso(min_date),
                "max_date": _iso(max_date),
            }
        except Exception as exc:
            result["staging"][table] = {"error": str(exc)}

    for collection in ["market_sequences", "market_sequences_xs", "model_runs", "benchmark_runs"]:
        try:
            result["curated"][collection] = {
                "document_count": db.mongo_db[collection].count_documents({})
            }
        except Exception as exc:
            result["curated"][collection] = {"error": str(exc)}

    try:
        latest_run = db.mongo_db.model_runs.find_one({}, {"_id": 0}, sort=[("created_at", -1)])
        result["ml"]["latest_model_run"] = latest_run
    except Exception as exc:
        result["ml"]["error"] = str(exc)

    try:
        latest_benchmark = db.mongo_db.benchmark_runs.find_one({}, {"_id": 0}, sort=[("created_at", -1)])
        result["benchmarks"]["latest_benchmark_run"] = latest_benchmark
    except Exception as exc:
        result["benchmarks"]["error"] = str(exc)

    for collection in ["ingestion_errors", "data_quality_logs"]:
        try:
            count = db.mongo_db[collection].count_documents({})
            latest_error = db.mongo_db[collection].find_one({}, {"_id": 0}, sort=[("created_at", -1)])
            result["errors"][collection] = {"count": count, "latest": latest_error}
        except Exception as exc:
            result["errors"][collection] = {"error": str(exc)}

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
