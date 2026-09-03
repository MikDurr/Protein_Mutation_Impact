"""Evaluation metrics shared by the regression and classification pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class RegressionMetrics:
    """Metrics for a continuous fitness-score prediction."""

    spearman: float
    pearson: float
    mse: float

    def as_dict(self) -> Dict[str, float]:
        return {"spearman": self.spearman, "pearson": self.pearson, "mse": self.mse}


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Score continuous predictions. Spearman correlation is the headline metric

    (it matches how ProteinGym itself ranks DMS models, and is robust to the
    fact that fitness scores are not on a meaningful linear scale).
    """
    spearman, _ = spearmanr(y_pred, y_true)
    pearson, _ = pearsonr(y_pred, y_true)
    mse = mean_squared_error(y_true, y_pred)
    return RegressionMetrics(spearman=float(spearman), pearson=float(pearson), mse=float(mse))


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics for a binary prediction at a fixed threshold, plus threshold-free scores."""

    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    auroc: float
    auprc: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "threshold": self.threshold,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "auroc": self.auroc,
            "auprc": self.auprc,
        }


def evaluate_classification(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> ClassificationMetrics:
    """Score binary predictions at `threshold`, plus threshold-independent AUROC/AUPRC."""
    y_pred = (y_proba >= threshold).astype(int)
    return ClassificationMetrics(
        threshold=float(threshold),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        auroc=float(roc_auc_score(y_true, y_proba)),
        auprc=float(average_precision_score(y_true, y_proba)),
    )


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return the classification threshold that maximizes F1 on (y_true, y_proba)."""
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1_scores))])
