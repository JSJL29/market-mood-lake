"""
Dataset PyTorch pour les séquences fenêtrées du S&P500 + VIX + Fear&Greed.

Split temporel (pas de shuffle aléatoire) : en finance, un split
train_test_split classique mélangerait le futur et le passé et
provoquerait une fuite d'information (data leakage). On découpe donc
train/val/test dans l'ordre chronologique.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MarketSequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_split(npz_path, train_ratio=0.7, val_ratio=0.15):
    """
    Charge le snapshot exporté depuis MongoDB et le découpe
    chronologiquement en train/val/test.
    """
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    data = np.load(npz_path, allow_pickle=False)
    features, labels = data["features"], data["labels"]

    if features.ndim != 3 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Expected features [samples, timesteps, features] and matching 1-D labels")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1)")
    if len(labels) < 3:
        raise ValueError("At least three samples are required for train/validation/test splits")

    split_metadata = {}
    if "dates" in data.files:
        dates = np.asarray(data["dates"]).astype(str)
        if dates.ndim != 1 or len(dates) != len(labels):
            raise ValueError("dates must be one-dimensional and match labels")
        unique_dates = np.unique(dates)
        train_date_end = int(len(unique_dates) * train_ratio)
        val_date_end = int(len(unique_dates) * (train_ratio + val_ratio))
        purge = features.shape[1] - 1
        if train_date_end <= purge or val_date_end - train_date_end <= purge or val_date_end >= len(unique_dates):
            raise ValueError("Not enough unique dates for purged train/validation/test splits")

        date_groups = {
            "train": unique_dates[:train_date_end - purge],
            "val": unique_dates[train_date_end:val_date_end - purge],
            "test": unique_dates[val_date_end:],
        }
        splits = {}
        for name, selected_dates in date_groups.items():
            mask = np.isin(dates, selected_dates)
            if not mask.any():
                raise ValueError(f"Empty {name} split after date grouping and purge")
            splits[name] = (features[mask], labels[mask])
        split_metadata = {
            "purge_dates": int(purge),
            "date_ranges": {
                name: {"start": str(values[0]), "end": str(values[-1]), "unique_dates": int(len(values))}
                for name, values in date_groups.items()
            },
        }
    else:
        n = len(labels)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        if train_end == 0 or val_end == train_end or val_end == n:
            raise ValueError("Split ratios produce an empty train, validation, or test partition")
        splits = {
            "train": (features[:train_end], labels[:train_end]),
            "val": (features[train_end:val_end], labels[train_end:val_end]),
            "test": (features[val_end:], labels[val_end:]),
        }

    # Normalisation : moyenne/écart-type calculés sur le train uniquement,
    # appliqués ensuite à val/test (pas de fuite d'information).
    train_feat = splits["train"][0]
    mean = train_feat.mean(axis=(0, 1), keepdims=True)
    std = train_feat.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    normalized = {}
    for name, (feat, lab) in splits.items():
        normalized[name] = ((feat - mean) / std, lab)

    feature_names = data["feature_names"].astype(str).tolist() if "feature_names" in data.files else None
    return normalized, {"mean": mean, "std": std, "feature_names": feature_names, **split_metadata}
