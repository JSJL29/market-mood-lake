import numpy as np
import pytest

from src.ml.dataset import load_and_split
from src.ml.train import majority_baseline, safe_auc, select_features


def test_temporal_split_uses_train_statistics_only(tmp_path):
    features = np.arange(20 * 2 * 7, dtype=float).reshape(20, 2, 7)
    labels = np.array([0, 1] * 10)
    path = tmp_path / "dataset.npz"
    np.savez(path, features=features, labels=labels)
    splits, stats = load_and_split(path)
    assert len(splits["train"][1]) == 14
    np.testing.assert_allclose(splits["train"][0].mean(axis=(0, 1)), 0, atol=1e-7)
    np.testing.assert_allclose(stats["mean"], features[:14].mean(axis=(0, 1), keepdims=True))


def test_invalid_or_empty_splits_are_rejected(tmp_path):
    path = tmp_path / "small.npz"
    np.savez(path, features=np.zeros((2, 3, 7)), labels=np.zeros(2))
    with pytest.raises(ValueError, match="three samples"):
        load_and_split(path)


def test_cross_sectional_split_groups_dates_and_purges_overlapping_windows(tmp_path):
    dates = np.repeat(np.array([f"2026-01-{day:02d}" for day in range(1, 31)], dtype="U10"), 2)
    features = np.zeros((len(dates), 3, 7), dtype=float)
    labels = np.arange(len(dates)) % 2
    path = tmp_path / "xs.npz"
    np.savez(path, features=features, labels=labels, dates=dates)
    splits, metadata = load_and_split(path)
    assert metadata["purge_dates"] == 2
    assert [len(splits[name][1]) for name in ("train", "val", "test")] == [38, 4, 10]
    assert metadata["date_ranges"]["train"]["end"] < metadata["date_ranges"]["val"]["start"]
    assert metadata["date_ranges"]["val"]["end"] < metadata["date_ranges"]["test"]["start"]


def test_feature_modes_and_metrics_are_stable():
    features = np.zeros((4, 3, 7))
    assert select_features(features, "price_only").shape == (4, 3, 4)
    assert select_features(features, "full").shape == (4, 3, 6)
    assert majority_baseline(np.array([0, 0, 0, 1])) == 0.75
    assert np.isnan(safe_auc([1, 1], [0.7, 0.8]))


def test_non_finite_features_and_non_binary_labels_are_rejected(tmp_path):
    bad_features = np.zeros((4, 2, 7))
    bad_features[0, 0, 0] = np.nan
    path = tmp_path / "bad.npz"
    np.savez(path, features=bad_features, labels=np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="NaN"):
        load_and_split(path)

    np.savez(path, features=np.zeros((4, 2, 7)), labels=np.array([0, 1, 2, 0]))
    with pytest.raises(ValueError, match="binary"):
        load_and_split(path)
