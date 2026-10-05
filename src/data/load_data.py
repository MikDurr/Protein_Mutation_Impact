"""Load the GFP deep mutational scan assay from the raw ProteinGym table."""
from __future__ import annotations

import re

import pandas as pd

from src.config import GFP_DMS_ID, PATHS

# Single mutation like "A23V", or several joined with ":" for multi-mutants
# like "K3R:V55A:Q94R".
_MUTANT_PATTERN = re.compile(r"^[A-Z]\d+[A-Z](:[A-Z]\d+[A-Z])*$")

REQUIRED_COLUMNS = ("mutant", "mutated_sequence", "target_seq", "DMS_score", "DMS_score_bin")


def load_gfp_dms() -> pd.DataFrame:
    """Load every valid GFP variant (single- and multi-mutant) with its scores."""
    df = pd.read_parquet(PATHS.raw_dms)
    gfp = df[df["DMS_id"] == GFP_DMS_ID].copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in gfp.columns]
    if missing:
        raise KeyError(f"GFP assay is missing expected columns: {missing}")

    gfp = gfp[gfp["mutant"].astype(str).str.match(_MUTANT_PATTERN)]
    gfp = gfp.dropna(subset=["DMS_score", "DMS_score_bin", "mutated_sequence"]).copy()
    gfp["DMS_score_bin"] = gfp["DMS_score_bin"].astype(int)

    return gfp.reset_index(drop=True)
