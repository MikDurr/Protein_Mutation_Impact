"""Generate a plain-text summary report of the trained models' performance."""
from __future__ import annotations

import json

from src.config import PATHS


def _load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def generate_report() -> None:
    """Write artifacts/reports/summary_report.txt from the saved metrics files."""
    feature_meta = _load_json(PATHS.features_dir / "metadata.json")
    regressor_metrics = _load_json(PATHS.models_dir / "regressor_metrics.json")
    classifier_metrics = _load_json(PATHS.models_dir / "classifier_metrics.json")

    lines = ["PROTEIN MUTATION IMPACT PREDICTION - SUMMARY REPORT", "=" * 60, ""]

    if feature_meta:
        lines += [
            "DATASET",
            "-" * 60,
            f"Variants: {feature_meta['n_samples']} (max {feature_meta['max_mutations_per_variant']} mutations/variant)",
            f"Class balance (tolerated/deleterious): {feature_meta['class_distribution']}",
            f"Embedding model: {feature_meta['model_name']} ({feature_meta['n_features']}-dim)",
            "",
        ]

    if regressor_metrics:
        zs, sup = regressor_metrics["zero_shot_only"], regressor_metrics["supervised_regressor"]
        lines += [
            "REGRESSION (primary model: predicts continuous DMS fitness score)",
            "-" * 60,
            f"Test set size: {regressor_metrics['n_test']}",
            f"Zero-shot ESM2 baseline:  Spearman={zs['spearman']:.4f}  Pearson={zs['pearson']:.4f}",
            f"Supervised RF regressor:  Spearman={sup['spearman']:.4f}  Pearson={sup['pearson']:.4f}",
            "",
        ]

    if classifier_metrics:
        lines += [
            "CLASSIFICATION (binary tolerated/deleterious, for comparison)",
            "-" * 60,
            f"AUROC: {classifier_metrics['auroc']:.4f}   AUPRC: {classifier_metrics['auprc']:.4f}",
            f"F1 @ threshold {classifier_metrics['threshold']:.3f}: {classifier_metrics['f1']:.4f}",
            "",
        ]

    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = PATHS.reports_dir / "summary_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"Saved report to {report_path}")


def main() -> None:
    generate_report()


if __name__ == "__main__":
    main()
