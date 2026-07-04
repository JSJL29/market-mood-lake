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
    data = np.load(npz_path, allow_pickle=True)
    features, labels = data["features"], data["labels"]

    n = len(labels)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

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

    return normalized, {"mean": mean, "std": std}
