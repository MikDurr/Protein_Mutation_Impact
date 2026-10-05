"""Build the GFP feature matrix: delta ESM2 embeddings + zero-shot scores + labels."""
from __future__ import annotations

import json

import numpy as np

from src.config import ESM2_MODEL_NAME, PATHS
from src.data.load_data import load_gfp_dms
from src.features.embeddings import delta_embeddings
from src.features.zero_shot import score_mutants


def main() -> None:
    """Load the assay, compute features for every variant, and save them to disk."""
    df = load_gfp_dms()
    wt_sequence = df["target_seq"].iloc[0]
    print(f"Loaded {len(df)} GFP variants")

    print("Computing delta ESM2 embeddings (mutant - wild-type)...")
    X = delta_embeddings(wt_sequence, df["mutated_sequence"].tolist())

    print("Computing zero-shot masked-marginal scores...")
    zero_shot = score_mutants(df["mutant"].tolist(), wt_sequence)

    y_binary = df["DMS_score_bin"].to_numpy(dtype=np.int64)
    n_mutations = df["mutant"].str.count(":").to_numpy(dtype=np.int64) + 1

    PATHS.features_dir.mkdir(parents=True, exist_ok=True)
    np.save(PATHS.features_dir / "X_delta.npy", X)
    np.save(PATHS.features_dir / "zero_shot_score.npy", zero_shot)
    np.save(PATHS.features_dir / "y_continuous.npy", df["DMS_score"].to_numpy(dtype=np.float32))
    np.save(PATHS.features_dir / "y_binary.npy", y_binary)
    np.save(PATHS.features_dir / "n_mutations.npy", n_mutations)

    metadata = {
        "n_samples": len(df),
        "n_features": int(X.shape[1]),
        "class_distribution": np.bincount(y_binary).tolist(),
        "model_name": ESM2_MODEL_NAME,
        "max_mutations_per_variant": int(n_mutations.max()),
    }
    (PATHS.features_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved features to {PATHS.features_dir} (X_delta: {X.shape})")


if __name__ == "__main__":
    main()
