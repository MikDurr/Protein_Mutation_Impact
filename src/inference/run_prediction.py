"""CLI: predict GFP fitness scores for specific mutations or the whole assay."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import PATHS
from src.data.load_data import load_gfp_dms
from src.inference.predict import predict_fitness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict GFP mutation fitness effects")
    mutation_group = parser.add_mutually_exclusive_group(required=True)
    mutation_group.add_argument("--mutations", type=str, nargs="+", help="Mutations to test, e.g. A23V S65T")
    mutation_group.add_argument("--all", action="store_true", help="Predict every variant in the assay")
    parser.add_argument("--wt_seq", type=str, default=None, help="Wild-type sequence (default: loaded from the assay)")
    parser.add_argument("--model", type=str, default=None, help="Path to a regressor bundle (default: the trained one)")
    parser.add_argument("--output", type=str, default=None, help="CSV path to save results (required with --all)")
    args = parser.parse_args()

    if args.all and not args.output:
        parser.error("--output is required when using --all")
    return args


def main() -> None:
    args = parse_args()

    df = load_gfp_dms()
    wt_seq = args.wt_seq or df["target_seq"].iloc[0]
    mutants = df["mutant"].tolist() if args.all else args.mutations

    kwargs = {"model_path": args.model} if args.model else {}
    result = predict_fitness(wt_seq, mutants, **kwargs)

    out_df = pd.DataFrame(
        {
            "mutant": result["mutants"],
            "predicted_dms_score": result["predicted_dms_score"],
            "zero_shot_score": result["zero_shot_score"],
        }
    ).sort_values("predicted_dms_score", ascending=False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, index=False)
        print(f"Saved {len(out_df)} predictions to {args.output}")
    else:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
