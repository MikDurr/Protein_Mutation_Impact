"""Mean-pooled ESM2 sequence embeddings, with an on-disk cache keyed by sequence."""
from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import torch
from tqdm import tqdm

from src.config import PATHS
from src.features.esm2_model import load_esm2


def _load_cache() -> Dict[str, np.ndarray]:
    if PATHS.esm2_cache.exists():
        return joblib.load(PATHS.esm2_cache)
    return {}


def _save_cache(cache: Dict[str, np.ndarray]) -> None:
    PATHS.esm2_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, PATHS.esm2_cache)


def _mean_pool(token_reps: torch.Tensor, tokens: torch.Tensor, alphabet) -> torch.Tensor:
    """Average token representations over real residues, excluding CLS/EOS/PAD."""
    is_residue = (
        (tokens != alphabet.padding_idx)
        & (tokens != alphabet.cls_idx)
        & (tokens != alphabet.eos_idx)
    )
    lengths = is_residue.sum(dim=1).clamp(min=1)
    masked = token_reps * is_residue.unsqueeze(-1)
    return masked.sum(dim=1) / lengths.unsqueeze(-1)


@torch.no_grad()
def embed_sequences(sequences: List[str], batch_size: int = 64) -> np.ndarray:
    """Return one mean-pooled ESM2 embedding per input sequence, in order.

    Sequences already in the on-disk cache are reused; only new sequences
    trigger a forward pass.
    """
    cache = _load_cache()
    to_compute = [s for s in dict.fromkeys(sequences) if s not in cache]

    if to_compute:
        model, alphabet, batch_converter, device = load_esm2()
        last_layer = model.num_layers

        for start in tqdm(range(0, len(to_compute), batch_size), desc="ESM2 embedding"):
            batch_seqs = to_compute[start : start + batch_size]
            _, _, tokens = batch_converter([(str(i), s) for i, s in enumerate(batch_seqs)])
            tokens = tokens.to(device)

            token_reps = model(tokens, repr_layers=[last_layer])["representations"][last_layer]
            pooled = _mean_pool(token_reps, tokens, alphabet).cpu().numpy().astype(np.float32)
            cache.update(zip(batch_seqs, pooled))

        _save_cache(cache)

    return np.stack([cache[s] for s in sequences]).astype(np.float32)


def delta_embeddings(wt_sequence: str, mutant_sequences: List[str], batch_size: int = 64) -> np.ndarray:
    """Embed the wild-type once and return (mutant - wild-type) for each mutant."""
    embeddings = embed_sequences([wt_sequence] + list(mutant_sequences), batch_size)
    return embeddings[1:] - embeddings[0]
