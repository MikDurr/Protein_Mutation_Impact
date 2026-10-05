"""Generate a plain-text summary report of the trained models' performance."""
from __future__ import annotations

import json

from src.config import PATHS


def main() -> None:
    """Write artifacts/reports/summary_report.txt from the saved metrics files."""
    meta = json.loads((PATHS.features_dir / "metadata.json").read_text())
    reg = json.loads((PATHS.models_dir / "regressor_metrics.json").read_text())
    clf = json.loads((PATHS.models_dir / "classifier_metrics.json").read_text())

    lines = [
        "PROTEIN MUTATION IMPACT PREDICTION - SUMMARY REPORT",
        "=" * 60,
        f"Variants: {meta['n_samples']} (up to {meta['max_mutations_per_variant']} mutations each)",
        f"Class balance (non-functional/functional): {meta['class_distribution']}",
        f"Embedding model: {meta['model_name']} ({meta['n_features']}-dim)",
        f"Train (subsampled) / test: {reg['n_train']} / {reg['n_test']}",
        "",
        f"Mutation-count baseline Spearman: {reg['mutation_count_baseline_spearman']:.4f}",
        f"Zero-shot ESM2 Spearman:          {reg['zero_shot_spearman']:.4f}",
        f"Supervised regressor Spearman:    {reg['regressor_spearman']:.4f}",
        f"Classifier ablation AUROC:        {clf['auroc']:.4f}",
        "",
        "Regressor Spearman within a fixed number of mutations:",
        *(f"  {k} mutation(s): {v:.4f}" for k, v in reg["regressor_spearman_by_n_mutations"].items()),
    ]

    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = PATHS.reports_dir / "summary_report.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
