"""Stratified train/test split of the saved feature matrix, and loading it back."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import MAX_TRAIN_SAMPLES, PATHS, RANDOM_STATE, TEST_SIZE

FEATURE_NAMES = ("X_delta", "zero_shot_score", "y_continuous", "y_binary", "n_mutations")


def load_split(name: str, part: str) -> np.ndarray:
    """Load one feature array for `part` ("train" or "test"), e.g. load_split("X_delta", "test")."""
    return np.load(PATHS.splits_dir / f"{name}_{part}.npy")


def train_subsample_indices(n_train: int) -> np.ndarray:
    """Fixed random subset of training rows, capped at MAX_TRAIN_SAMPLES for fit speed."""
    if n_train <= MAX_TRAIN_SAMPLES:
        return np.arange(n_train)
    rng = np.random.default_rng(RANDOM_STATE)
    return rng.choice(n_train, size=MAX_TRAIN_SAMPLES, replace=False)


def main() -> None:
    """Split all feature arrays with one shared index and save each half to disk."""
    arrays = {name: np.load(PATHS.features_dir / f"{name}.npy") for name in FEATURE_NAMES}
    train_idx, test_idx = train_test_split(
        np.arange(len(arrays["y_binary"])),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=arrays["y_binary"],
    )

    PATHS.splits_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(PATHS.splits_dir / f"{name}_train.npy", array[train_idx])
        np.save(PATHS.splits_dir / f"{name}_test.npy", array[test_idx])

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")


if __name__ == "__main__":
    main()
