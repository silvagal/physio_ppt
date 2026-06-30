#!/usr/bin/env python3
"""Generate publication-oriented figures for Strodthoff protocol results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_metric_curve(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns:
        return
    plt.figure(figsize=(7, 4))
    for model_name, g in df.groupby("model_name"):
        g = g.sort_values("fraction")
        x = g["fraction"].to_numpy()
        y = g[mean_col].to_numpy()
        yerr = g[std_col].to_numpy() if std_col in g.columns else None
        plt.plot(x, y, marker="o", label=str(model_name))
        if yerr is not None:
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    plt.xlabel("Label Fraction")
    plt.ylabel(metric.replace("test_", "").upper())
    plt.xscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_seed_stability(df: pd.DataFrame, metric_col: str, out_path: Path) -> None:
    if metric_col not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    df.boxplot(column=metric_col, by=["model_name", "fraction"], rot=45)
    plt.suptitle("")
    plt.title(f"Seed Stability: {metric_col}")
    plt.ylabel(metric_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser("Plot Strodthoff protocol results")
    parser.add_argument(
        "--summary_csv",
        default="physio_ppt/experiments/strodthoff_protocol/tables/strodthoff_summary_by_experiment.csv",
    )
    parser.add_argument(
        "--detailed_csv",
        default="physio_ppt/experiments/strodthoff_protocol/tables/strodthoff_runs_detailed.csv",
    )
    parser.add_argument("--output_dir", default="physio_ppt/experiments/strodthoff_protocol/figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summary_df = pd.read_csv(args.summary_csv)
    detailed_df = pd.read_csv(args.detailed_csv)

    _plot_metric_curve(summary_df, "test_macro_auroc", out_dir / "auroc_vs_label_fraction.png")
    _plot_metric_curve(summary_df, "test_macro_auprc", out_dir / "auprc_vs_label_fraction.png")
    _plot_metric_curve(summary_df, "test_macro_f1", out_dir / "f1_vs_label_fraction.png")
    _plot_seed_stability(detailed_df, "test_macro_auroc", out_dir / "seed_stability_macro_auroc.png")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()

