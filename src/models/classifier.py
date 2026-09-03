"""Train a binary (tolerated/deleterious) classifier, for comparison with the regressor.

Kept mainly to quantify how much of the original model's weakness was the
tiny single-mutant-only dataset (1,084 rows) versus the classification
framing itself: this trains the same delta-embedding representation on the
full multi-mutant dataset with SMOTE balancing.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import PATHS, RANDOM_STATE
from src.models.evaluate import evaluate_classification, find_best_threshold

MODEL_PATH = PATHS.models_dir / "rf_classifier_bundle.joblib"
METRICS_PATH = PATHS.models_dir / "classifier_metrics.json"

# See regressor.py: caps fit time to a few minutes on a laptop CPU.
MAX_TRAIN_SAMPLES = 15_000


def _load_split(name: str) -> np.ndarray:
    return np.load(PATHS.splits_dir / f"{name}.npy")


def train_and_evaluate() -> None:
    X_train, X_test = _load_split("X_train"), _load_split("X_test")
    y_train, y_test = _load_split("y_binary_train"), _load_split("y_binary_test")

    if len(X_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        subset = rng.choice(len(X_train), size=MAX_TRAIN_SAMPLES, replace=False)
        X_train, y_train = X_train[subset], y_train[subset]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"Training Random Forest classifier on {len(X_balanced)} balanced variants...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_balanced, y_balanced)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    threshold = find_best_threshold(y_test, y_proba)
    metrics = evaluate_classification(y_test, y_proba, threshold)

    print(f"AUROC={metrics.auroc:.4f}  AUPRC={metrics.auprc:.4f}  F1@{threshold:.3f}={metrics.f1:.4f}")

    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "model": model, "recommended_threshold": threshold},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics.as_dict(), indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
