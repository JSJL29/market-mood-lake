"""Export deterministic database content snapshots suitable for DVC versioning."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import mysql.connector
from pymongo import MongoClient


def snapshot_mysql(host: str, user: str, password: str, table: str, output: Path) -> int:
    connection = mysql.connector.connect(host=host, user=user, password=password, database="staging")
    cursor = connection.cursor(buffered=False)
    cursor.execute(f"SELECT * FROM `{table}` ORDER BY ticker, date")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                text = __import__("io").TextIOWrapper(zipped, encoding="utf-8", newline="")
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(cursor.column_names)
                for row in cursor:
                    writer.writerow(row)
                    count += 1
                text.flush()
                text.detach()
    finally:
        cursor.close()
        connection.close()
    return count


def snapshot_mongo(uri: str, collection: str, output: Path) -> int:
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    cursor = client.curated[collection].find({}, {"_id": 0}).sort([("ticker", 1), ("window_end_date", 1)])
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for document in cursor:
                    line = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
                    zipped.write(line.encode("utf-8") + b"\n")
                    count += 1
    finally:
        client.close()
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("mysql", "mongo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", default="market_data_xs")
    parser.add_argument("--collection", default="market_sequences_xs")
    parser.add_argument("--mysql-host", default="localhost")
    parser.add_argument("--mysql-user", default="root")
    parser.add_argument("--mysql-password", default="root")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    args = parser.parse_args()
    if args.kind == "mysql":
        count = snapshot_mysql(args.mysql_host, args.mysql_user, args.mysql_password, args.table, args.output)
    else:
        count = snapshot_mongo(args.mongo_uri, args.collection, args.output)
    print(json.dumps({"path": str(args.output), "rows": count, "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
