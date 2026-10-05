"""Generate evaluation plots for the trained regressor and classifier."""
from __future__ import annotations

import joblib
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import PATHS
from src.models import classifier, regressor
from src.models.split import load_split
from src.viz.plots import plot_regression_fit, plot_roc_curve


def main() -> None:
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    X_test = load_split("X_delta", "test")

    y_test = load_split("y_continuous", "test")
    y_pred = joblib.load(regressor.MODEL_PATH).predict(X_test)
    spearman = spearmanr(y_pred, y_test).statistic
    plot_regression_fit(y_test, y_pred, spearman, PATHS.figures_dir / "regression_fit.png")

    y_test_bin = load_split("y_binary", "test")
    y_proba = joblib.load(classifier.MODEL_PATH).predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test_bin, y_proba, roc_auc_score(y_test_bin, y_proba), PATHS.figures_dir / "classifier_roc.png")

    print(f"Saved plots to {PATHS.figures_dir}")


if __name__ == "__main__":
    main()
