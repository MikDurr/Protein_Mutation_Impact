"""CLI: run the full GFP mutation-impact pipeline end to end, in-process."""
from __future__ import annotations

from src.data.load_data import main as load_data
from src.features.build_features import main as build_features
from src.models.classifier import main as train_classifier
from src.models.regressor import main as train_regressor
from src.models.split import main as split_data
from src.models.validate import main as evaluate_models
from src.reporting.report import main as generate_report

STEPS = [
    ("Load GFP DMS assay", load_data),
    ("Build ESM2 delta-embedding + zero-shot features", build_features),
    ("Train/test split", split_data),
    ("Train fitness regressor (primary model)", train_regressor),
    ("Train classifier baseline (for comparison)", train_classifier),
    ("Generate evaluation plots", evaluate_models),
    ("Generate summary report", generate_report),
]


def main() -> None:
    for i, (description, step) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 70}\nSTEP {i}/{len(STEPS)}: {description}\n{'=' * 70}")
        step()
    print("\nPipeline complete. See artifacts/reports/summary_report.txt")


if __name__ == "__main__":
    main()
