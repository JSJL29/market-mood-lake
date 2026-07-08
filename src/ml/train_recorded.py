"""Train wrapper that records GRU metrics in MongoDB and a local JSON artifact.

This keeps the original `src.ml.train` usable for experiments while giving Airflow
an auditable ML step: export -> train -> write `curated.model_runs`.
"""
import argparse
import json
import os
import platform
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pymongo import MongoClient

from src.ml.dataset import load_and_split
from src.ml.train import run_training, select_features

def to_mongo_safe(value):
    """
    Convertit récursivement les objets non sérialisables BSON/JSON
    en types Python natifs compatibles MongoDB.
    """
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): to_mongo_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_mongo_safe(v) for v in value]

    return value

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _dataset_summary(splits: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for split_name, (features, labels) in splits.items():
        labels = np.asarray(labels)
        summary[split_name] = {
            "samples": int(len(labels)),
            "positive_rate": float(labels.mean()) if len(labels) else None,
            "feature_shape": list(features.shape),
        }
    return summary


def record_run(mongo_uri, db_name, collection, document):
    client = MongoClient(mongo_uri)
    db = client[db_name]

    safe_document = to_mongo_safe(document)
    db[collection].insert_one(safe_document)

    client.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU and record run metrics in MongoDB")
    parser.add_argument("--npz_path", default="data/curated_export/market_sequences.npz")
    parser.add_argument("--mode", choices=["price_only", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--mongo_uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    parser.add_argument("--mongo_db", default="curated")
    parser.add_argument("--mongo_collection", default="model_runs")
    parser.add_argument("--model_dir", default="models/model_runs")
    parser.add_argument("--run_note", default="")
    args = parser.parse_args()

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.mode}_{uuid.uuid4().hex[:8]}"
    splits, metadata = load_and_split(args.npz_path)
    dataset_summary = _dataset_summary(splits)

    train_feat, train_lab = splits["train"]
    val_feat, val_lab = splits["val"]
    test_feat, test_lab = splits["test"]

    train_feat = select_features(train_feat, args.mode)
    val_feat = select_features(val_feat, args.mode)
    test_feat = select_features(test_feat, args.mode)

    metrics = run_training(
        train_feat=train_feat,
        train_lab=train_lab,
        val_feat=val_feat,
        val_lab=val_lab,
        test_feat=test_feat,
        test_lab=test_lab,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr_patience=args.lr_patience,
        early_stop_patience=args.early_stop_patience,
        verbose=True,
    )

    now = datetime.now(timezone.utc)
    document = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "model": "GRU",
        "mode": args.mode,
        "npz_path": args.npz_path,
        "dataset_metadata": metadata,
        "dataset_summary": dataset_summary,
        "selected_feature_count": int(train_feat.shape[-1]),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr_patience": args.lr_patience,
            "early_stop_patience": args.early_stop_patience,
        },
        "metrics": {key: _jsonable(value) for key, value in metrics.items()},
        "environment": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "note": args.run_note,
    }

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    artifact_path = Path(args.model_dir) / f"{run_id}.json"
    artifact_path.write_text(json.dumps(document, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Run artifact écrit: {artifact_path}")

    record_run(args.mongo_uri, args.mongo_db, args.mongo_collection, document)
    print(f"Run enregistré dans MongoDB: {args.mongo_db}.{args.mongo_collection} / {run_id}")


if __name__ == "__main__":
    main()
