import numpy as np
import pytest

from src.ml.walk_forward import walk_forward


def test_walk_forward_rejects_invalid_fold_count(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, features=np.zeros((10, 3, 7)), labels=np.zeros(10))
    with pytest.raises(ValueError, match="n_folds"):
        walk_forward(path, "full", n_folds=1)


def test_walk_forward_rejects_invalid_validation_fraction(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, features=np.zeros((10, 3, 7)), labels=np.zeros(10))
    with pytest.raises(ValueError, match="val_fraction"):
        walk_forward(path, "full", n_folds=2, val_fraction=0)
