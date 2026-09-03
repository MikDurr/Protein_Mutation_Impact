"""Zero-shot mutation effect scoring via ESM2 masked-marginal log-odds.

This needs no labeled training data: for each wild-type position we mask
that residue and read off the model's log-probability for every amino
acid. A variant's score is the sum of (mutant - wild-type) log-odds over
its mutated positions -- the standard "masked marginal" method from
Meier et al. (2021), and a strong unsupervised baseline for DMS assays.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.config import ESM2_MODEL_NAME
from src.features.esm2_model import load_esm2
from src.mutations.parsing import Substitution, parse_mutant


@torch.no_grad()
def masked_marginal_log_probs(
    wt_sequence: str,
    model_name: str = ESM2_MODEL_NAME,
    batch_size: int = 8,
    device: str = "auto",
) -> np.ndarray:
    """Return log P(amino acid | WT with that position masked) for every position.

    Shape: (len(wt_sequence), alphabet_size). Computed once per wild-type
    sequence, independent of how many variants will be scored against it.
    """
    model, alphabet, batch_converter, resolved_device = load_esm2(model_name, device)

    _, _, wt_tokens = batch_converter([("wt", wt_sequence)])
    wt_tokens = wt_tokens.to(resolved_device)

    seq_len = len(wt_sequence)
    log_probs = np.zeros((seq_len, len(alphabet)), dtype=np.float32)

    for start in tqdm(range(0, seq_len, batch_size), desc="Masked-marginal scoring"):
        batch_positions = list(range(start, min(start + batch_size, seq_len)))
        batch_tokens = wt_tokens.repeat(len(batch_positions), 1).clone()
        for row, pos in enumerate(batch_positions):
            batch_tokens[row, pos + 1] = alphabet.mask_idx  # +1 to skip the CLS token

        logits = model(batch_tokens)["logits"]  # (B, T, V)
        batch_log_probs = torch.log_softmax(logits, dim=-1)

        for row, pos in enumerate(batch_positions):
            log_probs[pos] = batch_log_probs[row, pos + 1].cpu().numpy()

    return log_probs


def score_substitutions(substitutions: Sequence[Substitution], log_probs: np.ndarray, alphabet) -> float:
    """Sum per-position (mutant - wild-type) log-odds across a set of substitutions.

    Treats each substitution's effect as independent -- the standard
    approximation for scoring multi-mutants from single-position marginals.
    """
    total = 0.0
    for sub in substitutions:
        pos_idx = sub.position - 1
        total += (
            log_probs[pos_idx, alphabet.get_idx(sub.to_aa)]
            - log_probs[pos_idx, alphabet.get_idx(sub.from_aa)]
        )
    return float(total)


def score_mutants(
    mutants: Sequence[str],
    wt_sequence: str,
    model_name: str = ESM2_MODEL_NAME,
    batch_size: int = 8,
    device: str = "auto",
) -> np.ndarray:
    """Zero-shot log-odds score for each mutant string, relative to `wt_sequence`."""
    _, alphabet, _, _ = load_esm2(model_name, device)
    log_probs = masked_marginal_log_probs(wt_sequence, model_name, batch_size, device)
    return np.array(
        [score_substitutions(parse_mutant(m), log_probs, alphabet) for m in mutants],
        dtype=np.float32,
    )
