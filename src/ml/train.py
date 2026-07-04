"""
Entraîne un GRU binaire (hausse/baisse à horizon J+N) sur les séquences
fenêtrées S&P500 (+ indicateurs techniques + VIX/Fear&Greed) exportées
depuis MongoDB.

Permet de comparer deux configurations de features (--mode) pour évaluer
si le VIX / Fear&Greed apportent un gain par rapport aux seuls
indicateurs techniques calculés sur le prix. L'horizon du label (J+1,
J+5, ...) est fixé lors de la génération du curated (voir
preprocess_to_staging.py --horizon), pas ici.
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.ml.dataset import MarketSequenceDataset, load_and_split

FEATURE_INDEX = {
    "close": 0,
    "log_return": 1,
    "rolling_vol_20d": 2,
    "rsi_14": 3,
    "ma_ratio": 4,
    "vix_close": 5,
    "fear_greed_score": 6,
}

# Le niveau de prix brut ("close") est une série non-stationnaire : même
# normalisé, il dérive avec la tendance long terme du marché et peut
# introduire des corrélations parasites plutôt qu'un vrai signal
# prédictif. On entraîne donc uniquement sur des variables stationnaires
# (rendements, volatilité, indicateurs techniques, sentiment), jamais sur
# le niveau de prix lui-même.
TECHNICAL_FEATURES = ["log_return", "rolling_vol_20d", "rsi_14", "ma_ratio"]
SENTIMENT_FEATURES = ["vix_close", "fear_greed_score"]


class MarketGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        return out.squeeze(-1)


def select_features(features, mode):
    """
    mode='price_only' garde les indicateurs techniques calculés sur le
    prix (rendement, volatilité, RSI, ratio de moyennes mobiles), sans
    sentiment. 'full' ajoute vix_close/fear_greed_score. Le niveau de
    prix brut ("close") est exclu des deux modes : voir TECHNICAL_FEATURES.
    """
    if mode == "price_only":
        idx = [FEATURE_INDEX[f] for f in TECHNICAL_FEATURES]
    else:
        idx = [FEATURE_INDEX[f] for f in TECHNICAL_FEATURES + SENTIMENT_FEATURES]
    return features[:, :, idx]


def evaluate(model, loader):
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
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return acc, auc


def majority_baseline(labels):
    """Accuracy si on prédit toujours la classe majoritaire (référence honnête)."""
    positive_rate = labels.mean()
    return max(positive_rate, 1 - positive_rate)


def run_training(train_feat, train_lab, val_feat, val_lab, test_feat, test_lab, mode,
                  epochs=20, batch_size=32, lr=1e-3, hidden_size=32, num_layers=1, dropout=0.2,
                  lr_patience=3, early_stop_patience=5, verbose=True):
    """
    Entraîne et évalue un GRU à partir de tableaux déjà découpés en
    train/val/test (déjà sélectionnés selon `mode` et normalisés).
    Séparée de `train()` pour être réutilisable par une validation
    walk-forward (plusieurs découpages temporels successifs), pas
    seulement par le split unique 70/15/15 standard.

    Returns
    -------
    dict avec train_baseline, val_baseline, test_baseline, test_acc, test_auc
    """
    train_baseline = majority_baseline(train_lab)
    val_baseline = majority_baseline(val_lab)
    test_baseline = majority_baseline(test_lab)

    if verbose:
        print(f"[{mode}] Baseline classe majoritaire — train: {train_baseline:.4f} | "
              f"val: {val_baseline:.4f} | test: {test_baseline:.4f}")
        print(f"[{mode}] Hyperparamètres — hidden_size={hidden_size}, num_layers={num_layers}, "
              f"dropout={dropout}, lr={lr}, batch_size={batch_size}")

    train_loader = DataLoader(MarketSequenceDataset(train_feat, train_lab), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MarketSequenceDataset(val_feat, val_lab), batch_size=batch_size)
    test_loader = DataLoader(MarketSequenceDataset(test_feat, test_lab), batch_size=batch_size)

    model = MarketGRU(input_size=train_feat.shape[-1], hidden_size=hidden_size,
                       num_layers=num_layers, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience
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

        val_acc, val_auc = evaluate(model, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"[{mode}] Epoch {epoch + 1}/{epochs} - train_loss={train_loss:.4f} "
                  f"- val_loss={val_loss:.4f} - val_acc={val_acc:.4f} - val_auc={val_auc:.4f} - lr={current_lr:.2e}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"[{mode}] Early stopping à l'epoch {epoch + 1} "
                          f"(pas d'amélioration depuis {early_stop_patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_auc = evaluate(model, test_loader)
    verdict = "au-dessus" if test_acc > test_baseline else ("égal à" if test_acc == test_baseline else "en dessous de")
    if verbose:
        print(f"\n[{mode}] Résultat final sur le test set (meilleur modèle selon val_loss) : "
              f"acc={test_acc:.4f}, auc={test_auc:.4f} "
              f"({verdict} la baseline test de {test_baseline:.4f})")

    return {
        "train_baseline": train_baseline,
        "val_baseline": val_baseline,
        "test_baseline": test_baseline,
        "test_acc": test_acc,
        "test_auc": test_auc,
    }


def train(npz_path, mode, epochs, batch_size, lr, hidden_size=32, num_layers=1, dropout=0.2,
          lr_patience=3, early_stop_patience=5):
    splits, _ = load_and_split(npz_path)

    train_feat, train_lab = splits["train"]
    val_feat, val_lab = splits["val"]
    test_feat, test_lab = splits["test"]

    train_feat = select_features(train_feat, mode)
    val_feat = select_features(val_feat, mode)
    test_feat = select_features(test_feat, mode)

    run_training(train_feat, train_lab, val_feat, val_lab, test_feat, test_lab, mode,
                 epochs, batch_size, lr, hidden_size, num_layers, dropout,
                 lr_patience, early_stop_patience, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le GRU hausse/baisse S&P500")
    parser.add_argument("--npz_path", type=str, default="data/curated_export/market_sequences.npz")
    parser.add_argument("--mode", type=str, choices=["price_only", "full"], default="full",
                         help="price_only = indicateurs techniques seuls, full = + VIX + Fear&Greed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=32, help="Taille de l'état caché du GRU")
    parser.add_argument("--num_layers", type=int, default=1, help="Nombre de couches GRU empilées")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr_patience", type=int, default=3,
                         help="Epochs sans amélioration avant de réduire le learning rate")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                         help="Epochs sans amélioration avant l'arrêt anticipé")
    args = parser.parse_args()

    train(args.npz_path, args.mode, args.epochs, args.batch_size, args.lr,
          args.hidden_size, args.num_layers, args.dropout, args.lr_patience, args.early_stop_patience)