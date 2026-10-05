"""Train a binary (functional/non-functional) classifier as an ablation against the regressor.

Same features and same full dataset as the regressor, but the original
project's binary framing -- isolates how much of the original model's weakness
came from its tiny single-mutant-only dataset rather than the framing itself.
"""
from __future__ import annotations

import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import PATHS, RANDOM_STATE
from src.models.split import load_split, train_subsample_indices

MODEL_PATH = PATHS.models_dir / "rf_classifier.joblib"
METRICS_PATH = PATHS.models_dir / "classifier_metrics.json"


def main() -> None:
    idx = train_subsample_indices(len(load_split("y_binary", "train")))
    X_train, y_train = load_split("X_delta", "train")[idx], load_split("y_binary", "train")[idx]
    X_test, y_test = load_split("X_delta", "test"), load_split("y_binary", "test")

    print(f"Training Random Forest classifier on {len(y_train)} variants...")
    model = RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    auroc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print(f"Classifier AUROC={auroc:.4f}")

    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"auroc": auroc}, indent=2))


if __name__ == "__main__":
    main()
