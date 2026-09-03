"""Generate final evaluation plots for the trained regressor and classifier."""
from __future__ import annotations

import joblib
import numpy as np

from src.config import PATHS
from src.models.evaluate import evaluate_regression
from src.models.regressor import MODEL_PATH as REGRESSOR_PATH
from src.models.regressor import build_inputs
from src.viz.plots import plot_classifier_diagnostics, plot_regression_fit


def _load_split(name: str) -> np.ndarray:
    return np.load(PATHS.splits_dir / f"{name}.npy")


def evaluate_regressor() -> None:
    """Score the trained regressor on the held-out test set and plot its fit."""
    bundle = joblib.load(REGRESSOR_PATH)
    X_test, zs_test, y_test = (
        _load_split("X_test"),
        _load_split("zero_shot_score_test"),
        _load_split("y_continuous_test"),
    )
    inputs_scaled = bundle["scaler"].transform(build_inputs(X_test, zs_test))
    y_pred = bundle["model"].predict(inputs_scaled)
    metrics = evaluate_regression(y_test, y_pred)

    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_regression_fit(y_test, y_pred, metrics.spearman, PATHS.figures_dir / "regression_fit.png")
    print(f"Regressor test Spearman: {metrics.spearman:.4f}")


def evaluate_classifier() -> None:
    """Plot ROC/PR/probability diagnostics for the classifier baseline, if it was trained."""
    classifier_path = PATHS.models_dir / "rf_classifier_bundle.joblib"
    if not classifier_path.exists():
        print("No classifier bundle found, skipping classifier plots.")
        return

    bundle = joblib.load(classifier_path)
    X_test, y_test = _load_split("X_test"), _load_split("y_binary_test")
    X_test_scaled = bundle["scaler"].transform(X_test)
    y_proba = bundle["model"].predict_proba(X_test_scaled)[:, 1]

    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_classifier_diagnostics(
        y_test, y_proba, bundle["recommended_threshold"], PATHS.figures_dir / "classifier_diagnostics.png"
    )


def main() -> None:
    evaluate_regressor()
    evaluate_classifier()


if __name__ == "__main__":
    main()
