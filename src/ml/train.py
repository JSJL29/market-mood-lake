"""
Entraîne un GRU binaire hausse/baisse sur les séquences Market Mood Lake.

Modes disponibles :
- price_only : indicateurs techniques du ticker uniquement ;
- full : indicateurs techniques + VIX + Fear & Greed ;
- sector_full : full + nouvelle source sectorielle (ETF secteur + features relatives).

Le niveau de prix brut `close` reste exclu des features d'entraînement car il
est non stationnaire.
"""

from __future__ import annotations

import argparse
import copy
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

from src.ml.dataset import MarketSequenceDataset, load_and_split


LEGACY_FEATURE_INDEX = {
    "close": 0,
    "log_return": 1,
    "rolling_vol_20d": 2,
    "rsi_14": 3,
    "ma_ratio": 4,
    "vix_close": 5,
    "fear_greed_score": 6,
}

SECTOR_FEATURE_INDEX = {
    "close": 0,
    "log_return": 1,
    "rolling_vol_20d": 2,
    "rsi_14": 3,
    "ma_ratio": 4,
    "ticker_momentum_5d": 5,
    "vix_close": 6,
    "fear_greed_score": 7,
    "sector_log_return": 8,
    "sector_rolling_vol_20d": 9,
    "sector_momentum_5d": 10,
    "ticker_minus_sector_log_return": 11,
    "ticker_minus_sector_momentum_5d": 12,
}

TECHNICAL_FEATURES = [
    "log_return",
    "rolling_vol_20d",
    "rsi_14",
    "ma_ratio",
    "ticker_momentum_5d",
]
SENTIMENT_FEATURES = ["vix_close", "fear_greed_score"]
SECTOR_FEATURES = [
    "sector_log_return",
    "sector_rolling_vol_20d",
    "sector_momentum_5d",
    "ticker_minus_sector_log_return",
    "ticker_minus_sector_momentum_5d",
]


class MarketGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        return out.squeeze(-1)


def _feature_index_for(features: np.ndarray) -> dict[str, int]:
    n_features = features.shape[-1]
    if n_features >= len(SECTOR_FEATURE_INDEX):
        return SECTOR_FEATURE_INDEX
    if n_features == len(LEGACY_FEATURE_INDEX):
        return LEGACY_FEATURE_INDEX
    raise ValueError(
        f"Nombre de features inattendu dans le .npz : {n_features}. "
        "Regénère le curated/export après le remplacement des fichiers."
    )


def _names_for_mode(feature_index: dict[str, int], mode: str) -> list[str]:
    technical = [name for name in TECHNICAL_FEATURES if name in feature_index]
    if mode == "price_only":
        return technical
    if mode == "full":
        return technical + SENTIMENT_FEATURES
    if mode == "sector_full":
        missing = [name for name in SECTOR_FEATURES if name not in feature_index]
        if missing:
            raise ValueError(
                "mode='sector_full' demandé mais le snapshot .npz ne contient pas les features sectorielles. "
                "Relance preprocess_to_staging_xs -> process_to_curated_xs -> export_dataset."
            )
        return technical + SENTIMENT_FEATURES + SECTOR_FEATURES
    raise ValueError("mode doit valoir price_only, full ou sector_full")


def select_features(features: np.ndarray, mode: str) -> np.ndarray:
    feature_index = _feature_index_for(features)
    names = _names_for_mode(feature_index, mode)
    idx = [feature_index[name] for name in names]
    if mode == "sector_full":
        sector_idx = [feature_index[name] for name in SECTOR_FEATURES]
        if np.allclose(features[:, :, sector_idx], 0.0):
            raise ValueError("Features sectorielles absentes ou entièrement neutres; sector_full refusé")
    return features[:, :, idx].astype(np.float32)


def selected_feature_names(features: np.ndarray, mode: str) -> list[str]:
    feature_index = _feature_index_for(features)
    return _names_for_mode(feature_index, mode)


def expected_feature_schema(features: np.ndarray) -> list[str]:
    feature_index = _feature_index_for(features)
    return [name for name, _ in sorted(feature_index.items(), key=lambda item: item[1])]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def safe_auc(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true))
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def rmse_from_probs(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true).astype(float), np.asarray(y_prob).astype(float))))


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float, float]:
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)
            all_probs.extend(probs.tolist())
            all_labels.extend(y.tolist())

    acc = correct / total if total > 0 else 0.0
    auc = safe_auc(all_labels, all_probs)
    rmse = rmse_from_probs(all_labels, all_probs)
    return acc, auc, rmse


def majority_baseline(labels: np.ndarray) -> float:
    """Accuracy si on prédit toujours la classe majoritaire."""
    positive_rate = labels.mean()
    return max(positive_rate, 1 - positive_rate)


def fixed_class_accuracy(labels: np.ndarray, predicted_class: int) -> float:
    labels = np.asarray(labels).astype(int)
    return float(np.mean(labels == predicted_class)) if len(labels) else 0.0


def run_training(
    train_feat: np.ndarray,
    train_lab: np.ndarray,
    val_feat: np.ndarray,
    val_lab: np.ndarray,
    test_feat: np.ndarray,
    test_lab: np.ndarray,
    mode: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.2,
    lr_patience: int = 3,
    early_stop_patience: int = 5,
    verbose: bool = True,
    seed: int = 42,
    return_model: bool = False,
):
    set_reproducible_seed(seed)
    majority_class = int(np.asarray(train_lab).mean() >= 0.5)
    train_baseline = fixed_class_accuracy(train_lab, majority_class)
    val_baseline = fixed_class_accuracy(val_lab, majority_class)
    test_baseline = fixed_class_accuracy(test_lab, majority_class)

    if verbose:
        print(
            f"[{mode}] Baseline classe majoritaire — train: {train_baseline:.4f} | "
            f"val: {val_baseline:.4f} | test: {test_baseline:.4f}"
        )
        print(
            f"[{mode}] Hyperparamètres — hidden_size={hidden_size}, num_layers={num_layers}, "
            f"dropout={dropout}, lr={lr}, batch_size={batch_size}"
        )
        print(f"[{mode}] Shape train/val/test : {train_feat.shape} / {val_feat.shape} / {test_feat.shape}")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        MarketSequenceDataset(train_feat, train_lab), batch_size=batch_size,
        shuffle=True, generator=generator,
    )
    val_loader = DataLoader(MarketSequenceDataset(val_feat, val_lab), batch_size=batch_size)
    test_loader = DataLoader(MarketSequenceDataset(test_feat, test_lab), batch_size=batch_size)

    model = MarketGRU(
        input_size=train_feat.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X)
                val_loss += criterion(logits, y).item() * len(y)
        val_loss /= len(val_loader.dataset)

        val_acc, val_auc, val_rmse = evaluate(model, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(
                f"[{mode}] Epoch {epoch + 1}/{epochs} - train_loss={train_loss:.4f} "
                f"- val_loss={val_loss:.4f} - val_acc={val_acc:.4f} "
                f"- val_auc={val_auc:.4f} - val_rmse={val_rmse:.4f} - lr={current_lr:.2e}"
            )

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(
                        f"[{mode}] Early stopping à l'epoch {epoch + 1} "
                        f"(pas d'amélioration depuis {early_stop_patience} epochs)"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_auc, test_rmse = evaluate(model, test_loader)
    verdict = "au-dessus" if test_acc > test_baseline else ("égal à" if test_acc == test_baseline else "en dessous de")
    if verbose:
        print(
            f"\n[{mode}] Résultat final sur le test set (meilleur modèle selon val_loss) : "
            f"acc={test_acc:.4f}, auc={test_auc:.4f}, rmse={test_rmse:.4f} "
            f"({verdict} la baseline test de {test_baseline:.4f})"
        )

    metrics = {
        "train_baseline": float(train_baseline),
        "val_baseline": float(val_baseline),
        "test_baseline": float(test_baseline),
        "test_acc": float(test_acc),
        "test_auc": float(test_auc),
        "test_rmse": float(test_rmse),
        "seed": int(seed),
    }
    return (metrics, model) if return_model else metrics


def train(
    npz_path: str,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.2,
    lr_patience: int = 3,
    early_stop_patience: int = 5,
    seed: int = 42,
) -> None:
    splits, _ = load_and_split(npz_path)
    train_feat, train_lab = splits["train"]
    val_feat, val_lab = splits["val"]
    test_feat, test_lab = splits["test"]

    train_feat = select_features(train_feat, mode)
    val_feat = select_features(val_feat, mode)
    test_feat = select_features(test_feat, mode)

    run_training(
        train_feat,
        train_lab,
        val_feat,
        val_lab,
        test_feat,
        test_lab,
        mode,
        epochs,
        batch_size,
        lr,
        hidden_size,
        num_layers,
        dropout,
        lr_patience,
        early_stop_patience,
        verbose=True,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le GRU hausse/baisse Market Mood Lake")
    parser.add_argument("--npz_path", type=str, default="data/curated_export/market_sequences.npz")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["price_only", "full", "sector_full"],
        default="full",
        help="price_only = technique, full = + VIX/Fear&Greed, sector_full = + features sectorielles",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        args.npz_path,
        args.mode,
        args.epochs,
        args.batch_size,
        args.lr,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.lr_patience,
        args.early_stop_patience,
        args.seed,
    )
