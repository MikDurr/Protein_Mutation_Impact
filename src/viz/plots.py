"""Plotting functions for model evaluation and prediction results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_regression_fit(y_true: np.ndarray, y_pred: np.ndarray, spearman: float, save_path: Path) -> None:
    """Scatter predicted vs. actual DMS fitness score, annotated with Spearman rho."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, color="steelblue")
    ax.set_xlabel("Actual DMS score")
    ax.set_ylabel("Predicted DMS score")
    ax.set_title(f"Regressor fit on held-out test set (Spearman ρ = {spearman:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_classifier_diagnostics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, save_path: Path) -> None:
    """ROC curve, precision-recall curve, and probability histogram in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0].plot(fpr, tpr, linewidth=2, color="darkgreen")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1].plot(recall, precision, linewidth=2, color="darkblue")
    axes[1].axhline(y_true.mean(), color="red", linestyle="--", label=f"Baseline ({y_true.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].hist(y_proba[y_true == 0], bins=40, alpha=0.6, label="Tolerated", color="blue")
    axes[2].hist(y_proba[y_true == 1], bins=40, alpha=0.6, label="Deleterious", color="red")
    axes[2].axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.3f})")
    axes[2].set_xlabel("P(deleterious)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Predicted probability by class")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_position_sensitivity(predictions: pd.DataFrame, save_path: Path, min_variants_per_position: int = 3) -> None:
    """Mean predicted fitness score by mutated sequence position (single-mutants only)."""
    single = predictions[~predictions["mutant"].str.contains(":")].copy()
    single["position"] = single["mutant"].str[1:-1].astype(int)

    stats = single.groupby("position")["predicted_dms_score"].agg(["mean", "std", "count"]).reset_index()
    stats = stats[stats["count"] >= min_variants_per_position]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(stats["position"], stats["mean"], linewidth=2, color="darkblue")
    ax.fill_between(stats["position"], stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.25)
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Mean predicted DMS score")
    ax.set_title(f"Positional sensitivity (single mutants, ≥{min_variants_per_position} variants/position)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
