"""Load the shared ESM2 checkpoint onto the best available device."""
from __future__ import annotations

from functools import lru_cache

import torch

from src.config import ESM2_MODEL_NAME


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_esm2():
    """Return (model, alphabet, batch_converter, device), loaded once per process."""
    import esm  # fair-esm

    device = _best_device()
    model, alphabet = getattr(esm.pretrained, ESM2_MODEL_NAME)()
    model = model.eval().to(device)
    return model, alphabet, alphabet.get_batch_converter(), device
