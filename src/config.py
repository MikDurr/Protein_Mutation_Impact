"""Central path and constant configuration for the project.

Every other module imports paths from here instead of recomputing
`Path(__file__).resolve().parents[N]`, which was a recurring source of
bugs when scripts moved between directories.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used throughout the pipeline."""

    project_root: Path = PROJECT_ROOT
    raw_dms: Path = PROJECT_ROOT / "data" / "raw" / "proteingym_dms_substitutions.parquet"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    gfp_dataset: Path = PROJECT_ROOT / "data" / "processed" / "gfp_dms.parquet"

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    features_dir: Path = PROJECT_ROOT / "artifacts" / "features"
    splits_dir: Path = PROJECT_ROOT / "artifacts" / "train_test"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    figures_dir: Path = PROJECT_ROOT / "artifacts" / "figures"
    predictions_dir: Path = PROJECT_ROOT / "artifacts" / "predictions"
    reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"

    esm2_cache: Path = PROJECT_ROOT / "artifacts" / "esm2_cache.joblib"


PATHS = Paths()

# GFP assay identifier within the ProteinGym DMS substitutions table.
GFP_DMS_ID = "GFP_AEQVI_Sarkisyan_2016"

# ESM2 checkpoint used for both embeddings and zero-shot scoring.
ESM2_MODEL_NAME = "esm2_t12_35M_UR50D"

RANDOM_STATE = 42
TEST_SIZE = 0.2
