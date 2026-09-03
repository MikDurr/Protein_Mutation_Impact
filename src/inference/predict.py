"""Predict GFP fitness effects for arbitrary mutations using the trained regressor."""
from __future__ import annotations

from typing import Dict, Sequence

import joblib
import numpy as np

from src.features.embeddings import delta_embeddings
from src.features.zero_shot import score_mutants
from src.models.regressor import MODEL_PATH as REGRESSOR_PATH
from src.models.regressor import build_inputs
from src.mutations.parsing import apply_mutants


def predict_fitness(
    wt_sequence: str,
    mutants: Sequence[str],
    model_path=REGRESSOR_PATH,
    strict: bool = True,
) -> Dict[str, np.ndarray]:
    """Predict a continuous fitness score for each mutant string against `wt_sequence`.

    `mutants` may be single substitutions ("A23V") or multi-mutants
    ("K3R:V55A"), matching the format used throughout the assay.
    """
    bundle = joblib.load(model_path)
    scaler, model = bundle["scaler"], bundle["model"]

    mutant_sequences = apply_mutants(wt_sequence, mutants, strict=strict)
    X = delta_embeddings(wt_sequence, mutant_sequences)
    zero_shot_score = score_mutants(mutants, wt_sequence)

    inputs = build_inputs(X, zero_shot_score)
    inputs_scaled = scaler.transform(inputs)
    predicted_score = model.predict(inputs_scaled)

    return {
        "mutants": list(mutants),
        "mutant_sequences": mutant_sequences,
        "zero_shot_score": zero_shot_score,
        "predicted_dms_score": predicted_score,
    }
