"""Build the GFP feature matrix: delta ESM2 embeddings + zero-shot scores + labels.

Uses every valid variant in the assay (single- and multi-mutants), not just
the single substitutions used previously -- this is ~48x more labeled data
for the supervised model below.
"""
from __future__ import annotations

import json

import numpy as np

from src.config import ESM2_MODEL_NAME, PATHS
from src.data.load_data import load_gfp_dms
from src.features.embeddings import delta_embeddings
from src.features.zero_shot import score_mutants


def build_and_save() -> None:
    """Load the assay, compute features for every variant, and save them to disk."""
    df = load_gfp_dms()
    wt_sequence = df["target_seq"].iloc[0]
    mutant_sequences = df["mutated_sequence"].astype(str).tolist()

    print(f"Loaded {len(df)} variants ({df['mutated_sequence'].nunique()} unique sequences)")

    print("Computing delta ESM2 embeddings (mutant - wild-type)...")
    X = delta_embeddings(wt_sequence, mutant_sequences, batch_size=64)

    print("Computing zero-shot masked-marginal scores...")
    zero_shot = score_mutants(df["mutant"].tolist(), wt_sequence, batch_size=16)

    y_continuous = df["DMS_score"].to_numpy(dtype=np.float32)
    y_binary = df["DMS_score_bin"].to_numpy(dtype=np.int64)
    n_mutations = df["mutant"].str.count(":").add(1).to_numpy(dtype=np.int64)

    PATHS.features_dir.mkdir(parents=True, exist_ok=True)
    np.save(PATHS.features_dir / "X_delta.npy", X)
    np.save(PATHS.features_dir / "zero_shot_score.npy", zero_shot)
    np.save(PATHS.features_dir / "y_continuous.npy", y_continuous)
    np.save(PATHS.features_dir / "y_binary.npy", y_binary)
    np.save(PATHS.features_dir / "n_mutations.npy", n_mutations)
    df[["mutant"]].to_parquet(PATHS.features_dir / "mutant_ids.parquet", index=False)

    metadata = {
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_unique_sequences": int(df["mutated_sequence"].nunique()),
        "class_distribution": np.bincount(y_binary).tolist(),
        "model_name": ESM2_MODEL_NAME,
        "max_mutations_per_variant": int(n_mutations.max()),
    }
    (PATHS.features_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved features to {PATHS.features_dir}")
    print(f"  X_delta: {X.shape}")
    print(f"  class balance: {metadata['class_distribution']}")


def main() -> None:
    build_and_save()


if __name__ == "__main__":
    main()
