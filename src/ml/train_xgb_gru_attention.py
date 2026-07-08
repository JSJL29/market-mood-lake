"""
V4 sectorielle — Hybride XGBoost + GRU avec Attention pour Market Mood Lake.

Différence importante avec les versions précédentes :
- on n'entraîne pas XGBoost comme modèle séparé sur la fenêtre brute aplatie ;
- on entraîne d'abord un GRU+Attention ;
- on extrait son vecteur latent d'attention ;
- XGBoost apprend ensuite sur ce vecteur latent + quelques statistiques tabulaires simples.

À placer dans : src/ml/train_xgb_gru_attention.py
Lancement :
    python -m src.ml.train_xgb_gru_attention \
        --npz_path data/curated_export/market_sequences_xs.npz \
        --mode sector_full
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

from src.ml.dataset import MarketSequenceDataset, load_and_split

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost n'est pas installé. Fais : python -m pip install xgboost"
    ) from exc


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

# Même logique que train.py : ne pas utiliser le niveau de prix brut close.
TECHNICAL_FEATURES = ["log_return", "rolling_vol_20d", "rsi_14", "ma_ratio", "ticker_momentum_5d"]
SENTIMENT_FEATURES = ["vix_close", "fear_greed_score"]
SECTOR_FEATURES = [
    "sector_log_return",
    "sector_rolling_vol_20d",
    "sector_momentum_5d",
    "ticker_minus_sector_log_return",
    "ticker_minus_sector_momentum_5d",
]


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    auc: float
    rmse: float
    threshold: float


@dataclass(frozen=True)
class Result:
    mode: str
    train_shape: Tuple[int, int, int]
    val_shape: Tuple[int, int, int]
    test_shape: Tuple[int, int, int]
    train_baseline: float
    val_baseline: float
    test_baseline: float
    gru_attention_val: Metrics
    gru_attention_test: Metrics
    xgb_on_gru_attention_val: Metrics
    xgb_on_gru_attention_test: Metrics
    ensemble_weight_gru: float
    ensemble_val: Metrics
    ensemble_test: Metrics
    selected_model: str
    selected_val: Metrics
    selected_test: Metrics


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
    return features[:, :, idx].astype(np.float32)


def majority_baseline(labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if labels.size == 0:
        return 0.0
    positive_rate = float(labels.mean())
    return max(positive_rate, 1.0 - positive_rate)


def safe_auc(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def rmse_from_probs(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    """RMSE calculé entre le label binaire 0/1 et la probabilité prédite."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.sqrt(mean_squared_error(y_true, y_prob)))


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc=safe_auc(y_true, y_prob),
        rmse=rmse_from_probs(y_true, y_prob),
        threshold=float(threshold),
    )


def find_best_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    *,
    metric: str = "accuracy",
) -> Tuple[float, Metrics]:
    """Choisit le seuil uniquement sur la validation."""
    best_threshold = 0.5
    best_metrics = metrics_from_probs(y_val, p_val, threshold=0.5)
    best_score = (-1.0, -1.0)

    for threshold in np.linspace(0.05, 0.95, 91):
        m = metrics_from_probs(y_val, p_val, threshold=float(threshold))
        if metric == "accuracy":
            primary = m.accuracy
            secondary = m.auc if not np.isnan(m.auc) else -1.0
        elif metric == "auc":
            primary = m.auc if not np.isnan(m.auc) else -1.0
            secondary = m.accuracy
        else:
            raise ValueError("metric doit valoir 'accuracy' ou 'auc'")
        score = (primary, secondary)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = m

    return best_threshold, best_metrics


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        return context, weights


class MarketGRUAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.attention = AttentionPooling(out_size)
        self.norm = nn.LayerNorm(out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_size, 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.gru(x)
        context, attention_weights = self.attention(outputs)
        context = self.norm(context)
        return context, attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context, _ = self.encode(x)
        logits = self.fc(self.dropout(context)).squeeze(-1)
        return logits


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        MarketSequenceDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_probs_and_embeddings(
    model: MarketGRUAttention,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(features, labels, batch_size=batch_size, shuffle=False)
    probs, embeddings, attentions = [], [], []

    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            context, attn = model.encode(X)
            logits = model.fc(context).squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            embeddings.append(context.cpu().numpy())
            attentions.append(attn.cpu().numpy())

    return (
        np.concatenate(probs).astype(np.float32),
        np.concatenate(embeddings).astype(np.float32),
        np.concatenate(attentions).astype(np.float32),
    )


def train_gru_attention(
    train_feat: np.ndarray,
    train_lab: np.ndarray,
    val_feat: np.ndarray,
    val_lab: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr_patience: int,
    early_stop_patience: int,
    bidirectional: bool,
    use_pos_weight: bool,
    early_metric: str,
    device: torch.device,
) -> MarketGRUAttention:
    train_loader = make_loader(train_feat, train_lab, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_feat, val_lab, batch_size=batch_size, shuffle=False)

    model = MarketGRUAttention(
        input_size=train_feat.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    if use_pos_weight:
        pos = float(np.sum(train_lab == 1))
        neg = float(np.sum(train_lab == 0))
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
    )

    best_state = None
    best_score = float("inf") if early_metric == "val_loss" else -float("inf")
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item() * len(y)
        val_loss /= max(len(val_loader.dataset), 1)

        val_probs, _, _ = get_probs_and_embeddings(model, val_feat, val_lab, batch_size, device)
        _, val_metrics = find_best_threshold(val_lab, val_probs, metric="accuracy")
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[GRU+Attention] Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_auc={val_metrics.auc:.4f} "
            f"val_rmse={val_metrics.rmse:.4f} best_thr={val_metrics.threshold:.2f} lr={current_lr:.2e}"
        )

        scheduler.step(val_loss)

        if early_metric == "val_loss":
            score = val_loss
            improved = score < best_score
        elif early_metric == "val_auc":
            score = val_metrics.auc if not np.isnan(val_metrics.auc) else -1.0
            improved = score > best_score
        elif early_metric == "val_acc":
            score = val_metrics.accuracy
            improved = score > best_score
        else:
            raise ValueError("early_metric doit valoir val_loss, val_auc ou val_acc")

        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"[GRU+Attention] Early stopping à l'epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def sequence_summary_features(features: np.ndarray, attention_weights: np.ndarray | None = None) -> np.ndarray:
    """Features tabulaires compactes, pas de fenêtre aplatie complète."""
    last = features[:, -1, :]
    mean = features.mean(axis=1)
    std = features.std(axis=1)
    minimum = features.min(axis=1)
    maximum = features.max(axis=1)
    delta = features[:, -1, :] - features[:, 0, :]

    parts = [last, mean, std, minimum, maximum, delta]

    if attention_weights is not None:
        # Quelques statistiques sur l'attention : concentration et position moyenne.
        seq_len = attention_weights.shape[1]
        positions = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
        attn_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=1, keepdims=True)
        attn_max = attention_weights.max(axis=1, keepdims=True)
        attn_argmax = attention_weights.argmax(axis=1, keepdims=True).astype(np.float32) / max(seq_len - 1, 1)
        attn_center = np.sum(attention_weights * positions[None, :], axis=1, keepdims=True)
        parts.extend([attn_entropy, attn_max, attn_argmax, attn_center])

    return np.concatenate(parts, axis=1).astype(np.float32)


def make_xgb_meta_features(
    features: np.ndarray,
    gru_probs: np.ndarray,
    gru_embeddings: np.ndarray,
    attention_weights: np.ndarray,
    *,
    xgb_input: str,
) -> np.ndarray:
    tab = sequence_summary_features(features, attention_weights)
    p = gru_probs.reshape(-1, 1).astype(np.float32)

    if xgb_input == "embedding":
        X = np.concatenate([gru_embeddings, p], axis=1)
    elif xgb_input == "embedding_plus_stats":
        X = np.concatenate([gru_embeddings, p, tab], axis=1)
    elif xgb_input == "stats_only":
        X = tab
    else:
        raise ValueError("xgb_input doit valoir embedding, embedding_plus_stats ou stats_only")

    return X.astype(np.float32)


def train_xgb_on_gru_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    xgb_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    seed: int,
) -> XGBClassifier:
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = neg / max(pos, 1.0)

    model = XGBClassifier(
        n_estimators=xgb_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        min_child_weight=20,
        gamma=0.1,
        reg_lambda=10.0,
        reg_alpha=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train.astype(int), eval_set=[(X_val, y_val.astype(int))], verbose=False)
    return model


def choose_ensemble_weight(
    y_val: np.ndarray,
    gru_val_probs: np.ndarray,
    xgb_val_probs: np.ndarray,
    *,
    metric: str,
) -> Tuple[float, float, Metrics]:
    best_weight = 1.0
    best_threshold = 0.5
    best_metrics = Metrics(accuracy=-1.0, auc=float("nan"), rmse=float("inf"), threshold=0.5)
    best_score = (-1.0, -1.0)

    for weight in np.linspace(0.0, 1.0, 21):
        probs = weight * gru_val_probs + (1.0 - weight) * xgb_val_probs
        threshold, m = find_best_threshold(y_val, probs, metric=metric)
        if metric == "accuracy":
            score = (m.accuracy, m.auc if not np.isnan(m.auc) else -1.0)
        else:
            score = (m.auc if not np.isnan(m.auc) else -1.0, m.accuracy)
        if score > best_score:
            best_score = score
            best_weight = float(weight)
            best_threshold = float(threshold)
            best_metrics = m

    return best_weight, best_threshold, best_metrics


def select_final_model(
    *,
    val_baseline: float,
    test_baseline: float,
    gru_val: Metrics,
    gru_test: Metrics,
    xgb_val: Metrics,
    xgb_test: Metrics,
    ensemble_val: Metrics,
    ensemble_test: Metrics,
    min_improvement: float,
) -> Tuple[str, Metrics, Metrics]:
    baseline_val = Metrics(accuracy=val_baseline, auc=float("nan"), rmse=float("nan"), threshold=float("nan"))
    baseline_test = Metrics(accuracy=test_baseline, auc=float("nan"), rmse=float("nan"), threshold=float("nan"))

    candidates = {
        "baseline_majoritaire": (baseline_val, baseline_test),
        "gru_attention": (gru_val, gru_test),
        "xgb_on_gru_attention": (xgb_val, xgb_test),
        "ensemble": (ensemble_val, ensemble_test),
    }

    def score(m: Metrics) -> Tuple[float, float]:
        auc = m.auc if not np.isnan(m.auc) else -1.0
        return m.accuracy, auc

    best_name, (best_val, best_test) = max(candidates.items(), key=lambda item: score(item[1][0]))

    if best_name != "baseline_majoritaire" and best_val.accuracy < val_baseline + min_improvement:
        return "baseline_majoritaire", baseline_val, baseline_test

    return best_name, best_val, best_test


def run_training(args: argparse.Namespace) -> Result:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    splits, _ = load_and_split(args.npz_path)
    train_feat_raw, train_lab = splits["train"]
    val_feat_raw, val_lab = splits["val"]
    test_feat_raw, test_lab = splits["test"]

    train_feat = select_features(train_feat_raw, args.mode)
    val_feat = select_features(val_feat_raw, args.mode)
    test_feat = select_features(test_feat_raw, args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = majority_baseline(train_lab)
    val_base = majority_baseline(val_lab)
    test_base = majority_baseline(test_lab)

    print(f"Mode features : {args.mode}")
    print(f"Device PyTorch : {device}")
    print(f"Shape train/val/test : {train_feat.shape} / {val_feat.shape} / {test_feat.shape}")
    print(f"Baseline classe majoritaire — train={train_base:.4f} val={val_base:.4f} test={test_base:.4f}")

    gru = train_gru_attention(
        train_feat,
        train_lab,
        val_feat,
        val_lab,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr_patience=args.lr_patience,
        early_stop_patience=args.early_stop_patience,
        bidirectional=args.bidirectional,
        use_pos_weight=args.use_pos_weight,
        early_metric=args.early_metric,
        device=device,
    )

    train_gru_probs, train_emb, train_attn = get_probs_and_embeddings(gru, train_feat, train_lab, args.batch_size, device)
    val_gru_probs, val_emb, val_attn = get_probs_and_embeddings(gru, val_feat, val_lab, args.batch_size, device)
    test_gru_probs, test_emb, test_attn = get_probs_and_embeddings(gru, test_feat, test_lab, args.batch_size, device)

    gru_thr, gru_val_metrics = find_best_threshold(val_lab, val_gru_probs, metric=args.threshold_metric)
    gru_test_metrics = metrics_from_probs(test_lab, test_gru_probs, threshold=gru_thr)
    print(
        f"[GRU+Attention] thr={gru_thr:.2f} | "
        f"val_acc={gru_val_metrics.accuracy:.4f} val_auc={gru_val_metrics.auc:.4f} val_rmse={gru_val_metrics.rmse:.4f} | "
        f"test_acc={gru_test_metrics.accuracy:.4f} test_auc={gru_test_metrics.auc:.4f} test_rmse={gru_test_metrics.rmse:.4f}"
    )

    X_train_meta = make_xgb_meta_features(
        train_feat, train_gru_probs, train_emb, train_attn, xgb_input=args.xgb_input
    )
    X_val_meta = make_xgb_meta_features(
        val_feat, val_gru_probs, val_emb, val_attn, xgb_input=args.xgb_input
    )
    X_test_meta = make_xgb_meta_features(
        test_feat, test_gru_probs, test_emb, test_attn, xgb_input=args.xgb_input
    )

    xgb = train_xgb_on_gru_attention(
        X_train_meta,
        train_lab,
        X_val_meta,
        val_lab,
        xgb_estimators=args.xgb_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        seed=args.seed,
    )
    val_xgb_probs = xgb.predict_proba(X_val_meta)[:, 1]
    test_xgb_probs = xgb.predict_proba(X_test_meta)[:, 1]
    xgb_thr, xgb_val_metrics = find_best_threshold(val_lab, val_xgb_probs, metric=args.threshold_metric)
    xgb_test_metrics = metrics_from_probs(test_lab, test_xgb_probs, threshold=xgb_thr)
    print(
        f"[XGB sur embeddings GRU+Attention] thr={xgb_thr:.2f} | "
        f"val_acc={xgb_val_metrics.accuracy:.4f} val_auc={xgb_val_metrics.auc:.4f} val_rmse={xgb_val_metrics.rmse:.4f} | "
        f"test_acc={xgb_test_metrics.accuracy:.4f} test_auc={xgb_test_metrics.auc:.4f} test_rmse={xgb_test_metrics.rmse:.4f}"
    )

    w_gru, ens_thr, ensemble_val_metrics = choose_ensemble_weight(
        val_lab,
        val_gru_probs,
        val_xgb_probs,
        metric=args.threshold_metric,
    )
    ensemble_test_probs = w_gru * test_gru_probs + (1.0 - w_gru) * test_xgb_probs
    ensemble_test_metrics = metrics_from_probs(test_lab, ensemble_test_probs, threshold=ens_thr)
    print(
        f"[ENSEMBLE] poids_GRU={w_gru:.2f} poids_XGB={1.0 - w_gru:.2f} thr={ens_thr:.2f} | "
        f"val_acc={ensemble_val_metrics.accuracy:.4f} val_auc={ensemble_val_metrics.auc:.4f} val_rmse={ensemble_val_metrics.rmse:.4f} | "
        f"test_acc={ensemble_test_metrics.accuracy:.4f} test_auc={ensemble_test_metrics.auc:.4f} test_rmse={ensemble_test_metrics.rmse:.4f}"
    )

    selected_name, selected_val, selected_test = select_final_model(
        val_baseline=val_base,
        test_baseline=test_base,
        gru_val=gru_val_metrics,
        gru_test=gru_test_metrics,
        xgb_val=xgb_val_metrics,
        xgb_test=xgb_test_metrics,
        ensemble_val=ensemble_val_metrics,
        ensemble_test=ensemble_test_metrics,
        min_improvement=args.min_improvement,
    )
    print(
        f"[SELECTION] modèle retenu={selected_name} | "
        f"val_acc={selected_val.accuracy:.4f} val_auc={selected_val.auc:.4f} val_rmse={selected_val.rmse:.4f} | "
        f"test_acc={selected_test.accuracy:.4f} test_auc={selected_test.auc:.4f} test_rmse={selected_test.rmse:.4f}"
    )

    result = Result(
        mode=args.mode,
        train_shape=tuple(train_feat.shape),
        val_shape=tuple(val_feat.shape),
        test_shape=tuple(test_feat.shape),
        train_baseline=train_base,
        val_baseline=val_base,
        test_baseline=test_base,
        gru_attention_val=gru_val_metrics,
        gru_attention_test=gru_test_metrics,
        xgb_on_gru_attention_val=xgb_val_metrics,
        xgb_on_gru_attention_test=xgb_test_metrics,
        ensemble_weight_gru=w_gru,
        ensemble_val=ensemble_val_metrics,
        ensemble_test=ensemble_test_metrics,
        selected_model=selected_name,
        selected_val=selected_val,
        selected_test=selected_test,
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": gru.state_dict(),
                "input_size": train_feat.shape[-1],
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "bidirectional": args.bidirectional,
                "mode": args.mode,
                "xgb_input": args.xgb_input,
            },
            os.path.join(args.save_dir, "gru_attention.pt"),
        )
        xgb.save_model(os.path.join(args.save_dir, "xgboost_on_gru_attention.json"))
        with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        print(f"Modèles et métriques sauvegardés dans : {args.save_dir}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost sur embeddings GRU+Attention")
    parser.add_argument("--npz_path", type=str, default="data/curated_export/market_sequences.npz")
    parser.add_argument("--mode", type=str, choices=["price_only", "full", "sector_full"], default="full")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--early_metric", type=str, choices=["val_loss", "val_auc", "val_acc"], default="val_loss")

    parser.add_argument(
        "--xgb_input",
        type=str,
        choices=["embedding", "embedding_plus_stats", "stats_only"],
        default="embedding_plus_stats",
    )
    parser.add_argument("--xgb_estimators", type=int, default=150)
    parser.add_argument("--xgb_max_depth", type=int, default=2)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.7)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.7)

    parser.add_argument("--threshold_metric", type=str, choices=["accuracy", "auc"], default="accuracy")
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.005,
        help="Gain minimal en accuracy validation pour retenir autre chose que la baseline.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="models/xgb_gru_attention")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
