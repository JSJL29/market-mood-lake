"""
Validation walk-forward (rolling-origin) : au lieu d'un seul découpage
train/val/test, la chronologie est divisée en plusieurs segments
successifs. À chaque étape, le modèle est entraîné sur tout ce qui
précède un segment, puis évalué sur ce segment. Si le signal observé
dans train.py (single split) est réel plutôt qu'un artefact d'un split
chanceux, il doit apparaître de façon cohérente sur plusieurs segments,
pas seulement sur un seul.

Sert aussi à détecter une dérive de régime ("concept drift") : si la
performance varie énormément d'un segment à l'autre (parfois très
au-dessus de la baseline, parfois très en dessous), c'est le signe que
la relation entre les features et le label n'est pas stable dans le
temps, plutôt qu'un vrai signal exploitable.
"""
import argparse

import numpy as np

from src.ml.train import select_features, run_training


def load_raw_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["features"], data["labels"]


def normalize_with_train_stats(train_feat, *other_feats):
    """Normalise (z-score) à partir des stats du train uniquement, appliquées aux autres splits."""
    mean = train_feat.mean(axis=(0, 1), keepdims=True)
    std = train_feat.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    normalized_train = (train_feat - mean) / std
    normalized_others = [(feat - mean) / std for feat in other_feats]
    return (normalized_train, *normalized_others)


def walk_forward(npz_path, mode, n_folds=5, val_fraction=0.15, **train_kwargs):
    """
    Découpe les données (déjà triées chronologiquement lors de l'export)
    en `n_folds` segments contigus. Pour chaque segment k (à partir du
    2e), entraîne sur tous les segments précédents et teste sur le
    segment k. Une petite portion de la fin du train sert de validation
    (early stopping), comme dans le split standard.
    """
    features, labels = load_raw_npz(npz_path)
    n = len(labels)
    if n_folds < 2:
        raise ValueError("n_folds doit être supérieur ou égal à 2")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction doit être strictement comprise entre 0 et 1")
    if n < n_folds:
        raise ValueError("Pas assez de séquences pour le nombre de folds demandé")
    fold_size = n // n_folds

    print(f"[{mode}] Walk-forward sur {n} séquences, {n_folds} segments de ~{fold_size} chacun.\n")

    # Bornes des segments : le dernier segment absorbe le reste de la
    # division entière, pour ne rien perdre et éviter tout chevauchement.
    edges = [i * fold_size for i in range(n_folds)] + [n]

    fold_results = []

    for k in range(1, n_folds):
        train_end = edges[k]
        test_start = edges[k]
        test_end = edges[k + 1]

        val_size = max(1, int(train_end * val_fraction))
        actual_train_end = train_end - val_size
        if actual_train_end < 1 or test_end <= test_start:
            raise ValueError(f"Fold {k} vide avec les paramètres demandés")

        train_feat_raw = features[:actual_train_end]
        train_lab = labels[:actual_train_end]
        val_feat_raw = features[actual_train_end:train_end]
        val_lab = labels[actual_train_end:train_end]
        test_feat_raw = features[test_start:test_end]
        test_lab = labels[test_start:test_end]

        train_feat_raw = select_features(train_feat_raw, mode)
        val_feat_raw = select_features(val_feat_raw, mode)
        test_feat_raw = select_features(test_feat_raw, mode)

        train_feat, val_feat, test_feat = normalize_with_train_stats(train_feat_raw, val_feat_raw, test_feat_raw)

        print(f"--- Segment {k}/{n_folds - 1} : train=[0:{actual_train_end}] "
              f"val=[{actual_train_end}:{train_end}] test=[{test_start}:{test_end}] ---")

        result = run_training(train_feat, train_lab, val_feat, val_lab, test_feat, test_lab, mode,
                               verbose=False, **train_kwargs)
        result["fold"] = k
        fold_results.append(result)

        print(f"  test_acc={result['test_acc']:.4f} (baseline={result['test_baseline']:.4f}) "
              f"test_auc={result['test_auc']:.4f}\n")

    accs = [r["test_acc"] for r in fold_results]
    aucs = [r["test_auc"] for r in fold_results]
    baselines = [r["test_baseline"] for r in fold_results]
    diffs = [r["test_acc"] - r["test_baseline"] for r in fold_results]

    print("=" * 70)
    print(f"[{mode}] RÉSUMÉ WALK-FORWARD ({n_folds - 1} segments de test)")
    print("=" * 70)
    print(f"{'Segment':<10}{'Baseline':<12}{'Accuracy':<12}{'AUC':<12}{'Écart vs baseline':<20}")
    for r in fold_results:
        print(f"{r['fold']:<10}{r['test_baseline']:<12.4f}{r['test_acc']:<12.4f}"
              f"{r['test_auc']:<12.4f}{r['test_acc'] - r['test_baseline']:+.4f}")

    print(f"\nAccuracy moyenne  : {np.mean(accs):.4f} (écart-type {np.std(accs):.4f})")
    print(f"Baseline moyenne  : {np.mean(baselines):.4f}")
    print(f"AUC moyenne       : {np.mean(aucs):.4f} (écart-type {np.std(aucs):.4f})")
    print(f"Écart moyen accuracy - baseline : {np.mean(diffs):+.4f}")

    consistent_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    print(f"\nSigne de l'écart cohérent sur tous les segments : {'OUI' if consistent_sign else 'NON'}")
    if not consistent_sign:
        print("-> Le signe de l'écart change d'un segment à l'autre : signe de dérive de "
              "régime (la relation features/label n'est pas stable dans le temps), "
              "plutôt qu'un signal exploitable de façon fiable.")

    return fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation walk-forward du GRU hausse/baisse")
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["price_only", "full", "sector_full"], default="full")
    parser.add_argument("--n_folds", type=int, default=5, help="Nombre de segments temporels")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    args = parser.parse_args()

    walk_forward(
        args.npz_path, args.mode, args.n_folds,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
        lr_patience=args.lr_patience, early_stop_patience=args.early_stop_patience,
    )
