"""Stratified train/test split of the saved feature matrix."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PATHS, RANDOM_STATE, TEST_SIZE


def load_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (X_delta, zero_shot_score, y_continuous, y_binary) from artifacts/features/."""
    X = np.load(PATHS.features_dir / "X_delta.npy")
    zero_shot = np.load(PATHS.features_dir / "zero_shot_score.npy")
    y_continuous = np.load(PATHS.features_dir / "y_continuous.npy")
    y_binary = np.load(PATHS.features_dir / "y_binary.npy")
    return X, zero_shot, y_continuous, y_binary


def split_and_save() -> None:
    """Split all feature arrays with one shared index and save each half to disk."""
    X, zero_shot, y_continuous, y_binary = load_features()
    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_binary
    )

    PATHS.splits_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": X,
        "zero_shot_score": zero_shot,
        "y_continuous": y_continuous,
        "y_binary": y_binary,
    }
    for name, array in arrays.items():
        np.save(PATHS.splits_dir / f"{name}_train.npy", array[train_idx])
        np.save(PATHS.splits_dir / f"{name}_test.npy", array[test_idx])

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Positive rate (train/test): {y_binary[train_idx].mean():.3f} / {y_binary[test_idx].mean():.3f}")


def main() -> None:
    split_and_save()


if __name__ == "__main__":
    main()
