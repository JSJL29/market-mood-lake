from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import boto3
import json
import mysql.connector
from pymongo import MongoClient
from datetime import datetime

from src.api.routes_ingest import router as ingest_router

app = FastAPI(title="Market Mood Lake API")
app.include_router(ingest_router)


class DatabaseConnections:
    def __init__(self):
        # S3 (LocalStack)
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://localstack:4566'
        )

        # MySQL
        self.mysql_config = {
            'host': 'mysql',
            'user': 'root',
            'password': 'root',
            'database': 'staging'
        }

        # MongoDB
        self.mongo_uri = 'mongodb://mongodb:27017/'
        self.mongo_client = MongoClient(self.mongo_uri)
        self.mongo_db = self.mongo_client['curated']


db = DatabaseConnections()


@app.get("/raw/", response_model=List[dict])
async def get_raw_data(
    bucket: Optional[str] = Query("raw", description="Nom du bucket S3"),
    limit: Optional[int] = Query(10, description="Nombre maximum d'objets à lister"),
):
    """
    Liste les objets disponibles dans le bucket raw (S3) et retourne
    leurs métadonnées (clé, taille, date de dernière modification).
    """
    try:
        response = db.s3_client.list_objects_v2(Bucket=bucket)
        contents = response.get("Contents", [])[:limit]
        return [
            {
                "key": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            }
            for obj in contents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture depuis S3: {str(e)}")


@app.get("/staging/", response_model=List[dict])
async def get_staging_data(
    start_date: Optional[str] = Query(None, description="Date de début (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Date de fin (YYYY-MM-DD)"),
    limit: Optional[int] = Query(100, description="Nombre maximum de lignes"),
):
    """
    Récupère les données marché (S&P500 + VIX + Fear&Greed) depuis MySQL.
    """
    try:
        conn = mysql.connector.connect(**db.mysql_config)
        cursor = conn.cursor(dictionary=True)

        query = "SELECT * FROM market_data WHERE 1=1"
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

        cursor.close()
        conn.close()

        for row in rows:
            if 'date' in row and hasattr(row['date'], 'isoformat'):
                row['date'] = row['date'].isoformat()
            if 'created_at' in row and hasattr(row['created_at'], 'isoformat'):
                row['created_at'] = row['created_at'].isoformat()

        return rows

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture depuis MySQL: {str(e)}")


@app.get("/curated/", response_model=List[dict])
async def get_curated_data(
    limit: Optional[int] = Query(10, description="Nombre maximum de séquences à retourner"),
    label: Optional[int] = Query(None, description="Filtrer par label (1=hausse, 0=baisse)"),
):
    """
    Récupère les séquences fenêtrées prêtes pour le RNN depuis MongoDB.
    """
    try:
        collection = db.mongo_db.market_sequences

        query = {}
        if label is not None:
            query["label"] = label

        docs = list(collection.find(query, {'_id': 0}).limit(limit))
        return docs

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture depuis MongoDB: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Vérifie la santé de l'API et des connexions aux services (S3, MySQL, MongoDB).
    """
    status = {
        "api_status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": {}
    }

    try:
        db.s3_client.list_buckets()
        status["connections"]["s3"] = True
    except Exception:
        status["connections"]["s3"] = False

    try:
        conn = mysql.connector.connect(**db.mysql_config)
        conn.close()
        status["connections"]["mysql"] = True
    except Exception:
        status["connections"]["mysql"] = False

    try:
        db.mongo_client.server_info()
        status["connections"]["mongodb"] = True
    except Exception:
        status["connections"]["mongodb"] = False

    return status


@app.get("/stats")
async def stats():
    """
    Métriques sur le remplissage des buckets et bases de données.
    """
    result = {"timestamp": datetime.now().isoformat()}

    try:
        response = db.s3_client.list_objects_v2(Bucket="raw")
        result["raw_object_count"] = response.get("KeyCount", 0)
    except Exception as e:
        result["raw_object_count"] = f"error: {e}"

    try:
        conn = mysql.connector.connect(**db.mysql_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data")
        result["staging_row_count"] = cursor.fetchone()[0]
        cursor.close()
        conn.close()
    except Exception as e:
        result["staging_row_count"] = f"error: {e}"

    try:
        result["curated_document_count"] = db.mongo_db.market_sequences.count_documents({})
    except Exception as e:
        result["curated_document_count"] = f"error: {e}"

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
