"""CLI: predict GFP fitness scores for specific mutations or the whole assay."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.load_data import load_gfp_dms
from src.inference.predict import predict_fitness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict GFP mutation fitness effects")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mutations", nargs="+", help="Mutations to score, e.g. A206T S65T K3R:V55A")
    group.add_argument("--all", action="store_true", help="Score every variant in the assay")
    parser.add_argument("--output", help="CSV path to save results (required with --all)")
    args = parser.parse_args()
    if args.all and not args.output:
        parser.error("--output is required when using --all")
    return args


def main() -> None:
    args = parse_args()
    df = load_gfp_dms()
    mutants = df["mutant"].tolist() if args.all else args.mutations

    result = pd.DataFrame(predict_fitness(df["target_seq"].iloc[0], mutants))
    result = result.sort_values("predicted_dms_score", ascending=False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} predictions to {args.output}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
