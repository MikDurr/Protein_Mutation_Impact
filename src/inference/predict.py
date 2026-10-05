"""Predict GFP fitness for arbitrary mutations using the trained regressor."""
from __future__ import annotations

from typing import Dict, Sequence

import joblib
import numpy as np

from src.features.embeddings import delta_embeddings
from src.models.regressor import MODEL_PATH
from src.mutations.parsing import apply_mutants


def predict_fitness(wt_sequence: str, mutants: Sequence[str]) -> Dict[str, np.ndarray]:
    """Predict a fitness score for each mutant string ("A23V" or "K3R:V55A") against `wt_sequence`."""
    model = joblib.load(MODEL_PATH)
    X = delta_embeddings(wt_sequence, apply_mutants(wt_sequence, mutants))
    return {"mutant": list(mutants), "predicted_dms_score": model.predict(X)}
