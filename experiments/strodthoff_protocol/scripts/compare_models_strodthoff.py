#!/usr/bin/env python3
"""Quick ranking view by primary metric (Macro-AUROC)."""
from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser("Compare models under Strodthoff protocol")
    parser.add_argument(
        "--summary_csv",
        default="physio_ppt/experiments/strodthoff_protocol/tables/strodthoff_summary_by_experiment.csv",
    )
    parser.add_argument("--fraction", type=float, default=0.10, help="Label fraction to compare")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    subset = df[df["fraction"] == float(args.fraction)].copy()
    sort_col = "test_macro_auroc_mean"
    if sort_col not in subset.columns:
        raise KeyError(f"{sort_col} not found in {args.summary_csv}")
    subset = subset.sort_values(sort_col, ascending=False)
    cols = [c for c in ["experiment_name", "model_name", "fraction", "test_macro_auroc_mean", "test_macro_auprc_mean", "test_macro_f1_mean"] if c in subset.columns]
    print(subset[cols].to_string(index=False))


if __name__ == "__main__":
    main()

