"""Plot helpers for paper figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prep_out(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_low_label_curves(df: pd.DataFrame, metric: str, out_path: str | Path) -> None:
    """Plot low-label performance curves by method."""
    out = _prep_out(out_path)
    plt.figure(figsize=(7, 4))
    for method, group in df.groupby("method"):
        g = group.sort_values("label_fraction")
        plt.plot(g["label_fraction"], g[metric], marker="o", label=method)
    plt.xlabel("Label fraction")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_ablation_bars(df: pd.DataFrame, metric: str, out_path: str | Path) -> None:
    """Plot bar chart for ablation metrics."""
    out = _prep_out(out_path)
    agg = df.groupby("variant", as_index=False)[metric].mean().sort_values(metric, ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(agg["variant"], agg[metric])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_seed_delta(df: pd.DataFrame, out_path: str | Path) -> None:
    """Plot per-seed metric deltas against a reference method."""
    out = _prep_out(out_path)
    plt.figure(figsize=(7, 4))
    for method, group in df.groupby("method"):
        plt.plot(group["seed"], group["delta"], marker="o", label=method)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Seed")
    plt.ylabel("Delta")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_orderness_scatter(df: pd.DataFrame, out_path: str | Path) -> None:
    """Scatter plot between orderness score and downstream gain."""
    out = _prep_out(out_path)
    plt.figure(figsize=(5, 5))
    for method, group in df.groupby("method"):
        plt.scatter(group["orderness"], group["gain"], label=method, alpha=0.8)
    plt.xlabel("Orderness")
    plt.ylabel("Performance gain")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
