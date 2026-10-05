"""Parsing and application of mutation strings like "A23V" or "K3R:V55A"."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

_SINGLE_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass(frozen=True)
class Substitution:
    """A single amino-acid substitution at a 1-indexed sequence position."""

    from_aa: str
    position: int
    to_aa: str


def parse_substitution(token: str) -> Substitution:
    """Parse one substitution token, e.g. "A23V" -> Substitution('A', 23, 'V')."""
    match = _SINGLE_MUTATION.match(token.strip().upper())
    if not match:
        raise ValueError(f"Invalid mutation format: {token!r}. Expected e.g. 'A23V'")
    from_aa, pos_str, to_aa = match.groups()
    return Substitution(from_aa, int(pos_str), to_aa)


def parse_mutant(mutant: str) -> List[Substitution]:
    """Parse a possibly multi-mutant string, e.g. "K3R:V55A" -> two Substitutions."""
    return [parse_substitution(token) for token in mutant.split(":")]


def apply_mutant(wt_sequence: str, mutant: str) -> str:
    """Apply one or more ":"-joined substitutions to a wild-type sequence."""
    residues = list(wt_sequence)
    for sub in parse_mutant(mutant):
        idx = sub.position - 1
        if idx < 0 or idx >= len(residues):
            raise ValueError(f"Position {sub.position} out of range for sequence length {len(residues)}")
        if residues[idx] != sub.from_aa:
            raise ValueError(
                f"WT mismatch at position {sub.position}: expected '{sub.from_aa}', found '{residues[idx]}'"
            )
        residues[idx] = sub.to_aa
    return "".join(residues)


def apply_mutants(wt_sequence: str, mutants: Sequence[str]) -> List[str]:
    """Apply `apply_mutant` to each mutant string in `mutants`."""
    return [apply_mutant(wt_sequence, m) for m in mutants]
