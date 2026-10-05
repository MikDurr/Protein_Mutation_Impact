"""Plotting functions for model evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


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


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, auroc: float, save_path: Path) -> None:
    """ROC curve for the binary classifier ablation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, color="darkgreen", label=f"Classifier (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Classifier ROC curve (held-out test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
