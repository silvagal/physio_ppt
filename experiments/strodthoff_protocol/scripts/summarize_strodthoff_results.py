#!/usr/bin/env python3
"""Aggregate Strodthoff protocol runs into table-ready CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize Strodthoff protocol outputs")
    parser.add_argument(
        "--input_dir",
        default="physio_ppt/experiments/strodthoff_protocol/outputs/runs",
        help="Root directory containing run_summary.json files",
    )
    parser.add_argument(
        "--output_dir",
        default="physio_ppt/experiments/strodthoff_protocol/tables",
        help="Output directory for CSV summaries",
    )
    args = parser.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    for p in sorted(in_root.glob("**/run_summary.json")):
        row = _load_json(p)
        test_metrics_path = Path(str(row["test_metrics_json"]))
        if test_metrics_path.exists():
            tm = _load_json(test_metrics_path)
            for k, v in tm.get("test_metrics", {}).items():
                row[f"test_{k}"] = float(v)
        summaries.append(row)

    if not summaries:
        raise FileNotFoundError(f"No run_summary.json files found under {in_root}")

    detail_csv = out_root / "strodthoff_runs_detailed.csv"
    pd.DataFrame(summaries).to_csv(detail_csv, index=False)

    df = pd.DataFrame(summaries)
    metric_cols = [c for c in df.columns if c.startswith("test_")]
    agg_spec = {c: ["mean", "std"] for c in metric_cols}
    grouped = (
        df.groupby(["experiment_name", "model_name", "fraction", "use_pretrained"], as_index=False)
        .agg(agg_spec)
        .sort_values(["experiment_name", "fraction"])
    )
    grouped.columns = [
        "_".join([str(y) for y in x if y]).rstrip("_") if isinstance(x, tuple) else str(x) for x in grouped.columns
    ]
    summary_csv = out_root / "strodthoff_summary_by_experiment.csv"
    grouped.to_csv(summary_csv, index=False)

    paper_cols = [
        "experiment_name",
        "model_name",
        "fraction",
        "test_macro_auroc_mean",
        "test_macro_auroc_std",
        "test_macro_auprc_mean",
        "test_macro_f1_mean",
        "test_micro_f1_mean",
    ]
    paper_df = grouped[[c for c in paper_cols if c in grouped.columns]].copy()
    paper_csv = out_root / "strodthoff_table_paper_ready.csv"
    paper_df.to_csv(paper_csv, index=False)

    manifest = out_root / "strodthoff_summary_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["detail_csv", "summary_csv", "paper_csv", "num_runs"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerow(
            {
                "detail_csv": str(detail_csv),
                "summary_csv": str(summary_csv),
                "paper_csv": str(paper_csv),
                "num_runs": len(summaries),
            }
        )
    print(f"Saved summaries to {out_root}")


if __name__ == "__main__":
    main()

