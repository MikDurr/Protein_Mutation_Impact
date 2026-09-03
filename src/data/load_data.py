"""Load the GFP deep mutational scan assay from the raw ProteinGym table."""
from __future__ import annotations

import re

import pandas as pd

from src.config import GFP_DMS_ID, PATHS

# Single mutation like "A23V", or several joined with ":" for multi-mutants
# like "K3R:V55A:Q94R".
_MUTANT_PATTERN = re.compile(r"^[A-Z]\d+[A-Z](:[A-Z]\d+[A-Z])*$")

REQUIRED_COLUMNS = ("mutant", "mutated_sequence", "target_seq", "DMS_score", "DMS_score_bin")


def load_gfp_dms(raw_path=PATHS.raw_dms) -> pd.DataFrame:
    """Load every valid GFP variant (single- and multi-mutant) with its scores.

    Returns a DataFrame with the original ProteinGym columns, restricted to
    rows that have a well-formed `mutant` string and non-null scores.
    """
    df = pd.read_parquet(raw_path)
    gfp = df[df["DMS_id"] == GFP_DMS_ID].copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in gfp.columns]
    if missing:
        raise KeyError(f"GFP assay is missing expected columns: {missing}")

    gfp = gfp[gfp["mutant"].astype(str).str.match(_MUTANT_PATTERN)]
    gfp = gfp.dropna(subset=["DMS_score", "DMS_score_bin", "mutated_sequence"]).copy()
    gfp["DMS_score_bin"] = gfp["DMS_score_bin"].astype(int)

    return gfp.reset_index(drop=True)


def save_gfp_dms(df: pd.DataFrame, out_path=PATHS.gfp_dataset) -> None:
    """Write the loaded GFP assay to `data/processed/` for downstream steps."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    """CLI entrypoint: load the raw assay and cache it under data/processed/."""
    df = load_gfp_dms()
    save_gfp_dms(df)
    print(f"Loaded {len(df)} GFP variants ({df['mutant'].str.count(':').add(1).max()} max mutations/variant)")
    print(f"Label balance (DMS_score_bin): {df['DMS_score_bin'].value_counts().to_dict()}")
    print(f"Saved to: {PATHS.gfp_dataset}")


if __name__ == "__main__":
    main()
