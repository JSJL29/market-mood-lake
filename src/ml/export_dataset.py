"""
Exporte une collection curated (market_sequences ou market_sequences_xs)
depuis MongoDB vers un snapshot local (.npz), consommé ensuite par
train.py.

On évite de brancher le DataLoader PyTorch directement sur MongoDB :
le volume de données peut être important en cross-sectional (jusqu'à
plusieurs millions de séquences), et un snapshot local évite une
dépendance réseau répétée à chaque epoch.

À grande échelle, on évite aussi de matérialiser tous les documents dans
une liste Python (`list(collection.find(...))`) avant de construire les
tableaux NumPy : ça doublerait la mémoire nécessaire (liste de dicts +
tableau final). On préalloue directement les tableaux NumPy à la bonne
taille, puis on les remplit en itérant le curseur MongoDB un document à
la fois.
"""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from pymongo import MongoClient


def export_dataset(mongo_uri, output_path, collection_name="market_sequences", progress_every=200_000):
    client = MongoClient(mongo_uri)
    try:
        collection = client.curated[collection_name]

        count = collection.count_documents({})
        if count == 0:
            raise RuntimeError(f"Aucun document trouvé dans curated.{collection_name}")

    # Récupère la forme des features (window_size, n_features) depuis un
    # premier document, pour préallouer le tableau final.
        sample = collection.find_one({}, {"_id": 0, "features": 1, "metadata.feature_columns": 1})
        if not sample or not sample.get("features") or not sample["features"][0]:
            raise ValueError(f"Document invalide dans curated.{collection_name}: features absentes")
        window_size = len(sample["features"])
        n_features = len(sample["features"][0])
        feature_names = sample.get("metadata", {}).get("feature_columns")
        if not feature_names or len(feature_names) != n_features:
            raise ValueError("metadata.feature_columns absent ou incohérent dans la collection Curated")

        print(f"{count} documents à exporter, forme des features par séquence : ({window_size}, {n_features})")

        features = np.empty((count, window_size, n_features), dtype=np.float32)
        labels = np.empty(count, dtype=np.int64)
        dates = np.empty(count, dtype="U64")
        tickers = np.empty(count, dtype="U32")

    # Tri global par date de fin de fenêtre : essentiel pour que le split
    # temporel (train/val/test) reste chronologiquement cohérent même
    # lorsque plusieurs tickers partagent les mêmes dates (cross-sectional).
        cursor = collection.find({}, {"_id": 0}).sort("window_end_date", 1).batch_size(2000)

        exported_count = 0
        for i, doc in enumerate(cursor):
            if i >= count:
                raise RuntimeError("La collection a changé pendant l'export (documents ajoutés)")
            if np.shape(doc.get("features")) != (window_size, n_features):
                raise ValueError(f"Shape de features incohérente au document {i}")
            features[i] = doc["features"]
            labels[i] = doc["label"]
            dates[i] = doc["window_end_date"]
            tickers[i] = doc.get("ticker", "SP500")
            exported_count = i + 1

            if (i + 1) % progress_every == 0:
                print(f"  {i + 1}/{count} documents traités...")

        if exported_count != count:
            raise RuntimeError(
                f"La collection a changé pendant l'export: attendu={count}, parcouru={exported_count}"
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
            temporary_path = Path(handle.name)
        try:
            np.savez(
                temporary_path, features=features, labels=labels, dates=dates, tickers=tickers,
                feature_names=np.asarray(feature_names, dtype="U64"),
            )
            os.replace(temporary_path, output)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"Export terminé : {count} séquences -> {output}")
        print(f"Shape features: {features.shape}, shape labels: {labels.shape}, "
              f"tickers uniques: {len(set(tickers.tolist()))}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MongoDB curated -> snapshot local pour l'entraînement")
    parser.add_argument("--mongo_uri", type=str, default="mongodb://localhost:27017/")
    parser.add_argument("--output_path", type=str, default="data/curated_export/market_sequences.npz")
    parser.add_argument("--collection", type=str, default="market_sequences",
                         choices=["market_sequences", "market_sequences_xs"],
                         help="market_sequences = indice unique, market_sequences_xs = coupe transversale multi-actions")
    args = parser.parse_args()

    export_dataset(args.mongo_uri, args.output_path, args.collection)
