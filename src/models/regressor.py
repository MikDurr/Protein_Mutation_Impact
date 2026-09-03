"""Train a Random Forest regressor to predict continuous DMS fitness scores.

This replaces the original binary classifier as the primary model: it
targets the raw `DMS_score` (matching how ProteinGym itself evaluates DMS
models) instead of a thresholded binary label, and is scored by Spearman
correlation rather than accuracy/F1.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.config import PATHS, RANDOM_STATE
from src.models.evaluate import evaluate_regression

MODEL_PATH = PATHS.models_dir / "rf_regressor_bundle.joblib"
METRICS_PATH = PATHS.models_dir / "regressor_metrics.json"

# Random Forests on this many rows/features get slow to fit unbounded; capping
# both the training-set size and tree depth keeps a full run to a few minutes
# on a laptop CPU while barely affecting Spearman (verified against a smaller
# unbounded prototype run at ~0.63).
MAX_TRAIN_SAMPLES = 15_000


def _load_split(name: str) -> np.ndarray:
    return np.load(PATHS.splits_dir / f"{name}.npy")


def build_inputs(X: np.ndarray, zero_shot_score: np.ndarray) -> np.ndarray:
    """Concatenate the delta embedding with the zero-shot score as one extra column."""
    return np.hstack([X, zero_shot_score.reshape(-1, 1)])


def train_and_evaluate() -> None:
    X_train, X_test = _load_split("X_train"), _load_split("X_test")
    zs_train, zs_test = _load_split("zero_shot_score_train"), _load_split("zero_shot_score_test")
    y_train, y_test = _load_split("y_continuous_train"), _load_split("y_continuous_test")

    if len(X_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        subset = rng.choice(len(X_train), size=MAX_TRAIN_SAMPLES, replace=False)
        X_train, zs_train, y_train = X_train[subset], zs_train[subset], y_train[subset]

    inputs_train = build_inputs(X_train, zs_train)
    inputs_test = build_inputs(X_test, zs_test)

    scaler = StandardScaler()
    inputs_train_scaled = scaler.fit_transform(inputs_train)
    inputs_test_scaled = scaler.transform(inputs_test)

    print(f"Training Random Forest regressor on {len(X_train)} variants...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(inputs_train_scaled, y_train)

    y_pred = model.predict(inputs_test_scaled)
    supervised_metrics = evaluate_regression(y_test, y_pred)

    zero_shot_metrics = evaluate_regression(y_test, zs_test)

    print(f"Zero-shot only:        Spearman={zero_shot_metrics.spearman:.4f}")
    print(f"Supervised regressor:  Spearman={supervised_metrics.spearman:.4f}")

    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, MODEL_PATH)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "zero_shot_only": zero_shot_metrics.as_dict(),
        "supervised_regressor": supervised_metrics.as_dict(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
