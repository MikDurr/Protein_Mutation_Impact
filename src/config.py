"""Central path and constant configuration for the project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used throughout the pipeline."""

    raw_dms: Path = PROJECT_ROOT / "data" / "raw" / "proteingym_dms_substitutions.parquet"
    features_dir: Path = PROJECT_ROOT / "artifacts" / "features"
    splits_dir: Path = PROJECT_ROOT / "artifacts" / "train_test"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    figures_dir: Path = PROJECT_ROOT / "artifacts" / "figures"
    reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"
    esm2_cache: Path = PROJECT_ROOT / "artifacts" / "esm2_cache.joblib"


PATHS = Paths()

# GFP assay identifier within the ProteinGym DMS substitutions table.
GFP_DMS_ID = "GFP_AEQVI_Sarkisyan_2016"

# ESM2 checkpoint used for both embeddings and zero-shot scoring.
ESM2_MODEL_NAME = "esm2_t12_35M_UR50D"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forests on all ~41k training rows take 30+ minutes on a laptop CPU;
# a 15k random subsample fits in a few minutes with the same test Spearman.
MAX_TRAIN_SAMPLES = 15_000
