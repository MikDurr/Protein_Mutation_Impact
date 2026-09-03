"""Load the shared ESM2 checkpoint and resolve the compute device.

Kept separate from embedding/scoring logic so both `embeddings.py` and
`zero_shot.py` load the exact same model instance shape without duplicating
the fair-esm loading boilerplate.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch

from src.config import ESM2_MODEL_NAME


def resolve_device(preferred: str = "auto") -> torch.device:
    """Pick the best available torch device, or honor an explicit choice."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=4)
def load_esm2(model_name: str = ESM2_MODEL_NAME, device: str = "auto"):
    """Load an ESM2 model + alphabet + batch converter onto `device`.

    Cached per (model_name, device) so repeated calls within a process
    reuse the same loaded weights instead of re-reading them from disk.
    """
    import esm  # fair-esm

    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"Unknown ESM2 model_name={model_name!r}")

    resolved_device = resolve_device(device)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(resolved_device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, resolved_device


def last_layer_index(model) -> int:
    """ESM2 representation layers are numbered 1..num_layers; return the last."""
    return model.num_layers
