"""Mean-pooled ESM2 sequence embeddings, with an on-disk cache keyed by sequence."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from tqdm import tqdm

from src.config import ESM2_MODEL_NAME, PATHS
from src.features.esm2_model import last_layer_index, load_esm2


def _load_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        obj = joblib.load(cache_path)
        if isinstance(obj, dict):
            return obj
    return {}


def _save_cache(cache: Dict[str, np.ndarray], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, cache_path)


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
def embed_sequences(
    sequences: List[str],
    model_name: str = ESM2_MODEL_NAME,
    batch_size: int = 64,
    device: str = "auto",
    cache_path: Path = PATHS.esm2_cache,
) -> np.ndarray:
    """Return one mean-pooled ESM2 embedding per input sequence, in order.

    Sequences already present in the on-disk cache are reused; only new
    sequences trigger a forward pass.
    """
    cache = _load_cache(cache_path)
    to_compute = [s for s in dict.fromkeys(sequences) if s not in cache]

    if to_compute:
        model, alphabet, batch_converter, resolved_device = load_esm2(model_name, device)
        repr_layer = last_layer_index(model)

        for start in tqdm(range(0, len(to_compute), batch_size), desc="ESM2 embedding"):
            batch_seqs = to_compute[start : start + batch_size]
            batch = [(f"seq_{start + i}", s) for i, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(resolved_device)

            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_reps = out["representations"][repr_layer]
            pooled = _mean_pool(token_reps, tokens, alphabet).cpu().numpy().astype(np.float32)

            for seq, vec in zip(batch_seqs, pooled):
                cache[seq] = vec

        _save_cache(cache, cache_path)

    return np.stack([cache[s] for s in sequences], axis=0).astype(np.float32)


def delta_embeddings(wt_sequence: str, mutant_sequences: List[str], **embed_kwargs) -> np.ndarray:
    """Embed the wild-type once and return (mutant - wild-type) for each mutant."""
    all_embeddings = embed_sequences([wt_sequence] + list(mutant_sequences), **embed_kwargs)
    wt_embedding, mutant_embeddings = all_embeddings[0], all_embeddings[1:]
    return mutant_embeddings - wt_embedding[None, :]
