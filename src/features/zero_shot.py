"""Zero-shot mutation effect scoring via ESM2 masked-marginal log-odds.

Needs no labeled data: for each wild-type position we mask that residue and
read off the model's log-probability for every amino acid. A variant's score
is the sum of (mutant - wild-type) log-odds over its mutated positions -- the
"masked marginal" method from Meier et al. (2021).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.features.esm2_model import load_esm2
from src.mutations.parsing import parse_mutant


@torch.no_grad()
def masked_marginal_log_probs(wt_sequence: str, batch_size: int = 16) -> np.ndarray:
    """Return log P(amino acid | WT with that position masked) for every position.

    Shape: (len(wt_sequence), alphabet_size). Computed once per wild-type,
    regardless of how many variants are scored against it.
    """
    model, alphabet, batch_converter, device = load_esm2()
    _, _, wt_tokens = batch_converter([("wt", wt_sequence)])
    wt_tokens = wt_tokens.to(device)

    seq_len = len(wt_sequence)
    log_probs = np.zeros((seq_len, len(alphabet)), dtype=np.float32)

    for start in tqdm(range(0, seq_len, batch_size), desc="Masked-marginal scoring"):
        positions = range(start, min(start + batch_size, seq_len))
        batch_tokens = wt_tokens.repeat(len(positions), 1)
        for row, pos in enumerate(positions):
            batch_tokens[row, pos + 1] = alphabet.mask_idx  # +1 skips the CLS token

        batch_log_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
        for row, pos in enumerate(positions):
            log_probs[pos] = batch_log_probs[row, pos + 1].cpu().numpy()

    return log_probs


def score_mutants(mutants: Sequence[str], wt_sequence: str) -> np.ndarray:
    """Zero-shot log-odds score for each mutant string, relative to `wt_sequence`.

    Multi-mutant scores sum the per-position log-odds, treating each
    substitution as independent.
    """
    _, alphabet, _, _ = load_esm2()
    log_probs = masked_marginal_log_probs(wt_sequence)

    def score(mutant: str) -> float:
        return float(sum(
            log_probs[sub.position - 1, alphabet.get_idx(sub.to_aa)]
            - log_probs[sub.position - 1, alphabet.get_idx(sub.from_aa)]
            for sub in parse_mutant(mutant)
        ))

    return np.array([score(m) for m in mutants], dtype=np.float32)
