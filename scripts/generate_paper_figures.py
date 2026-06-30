#!/usr/bin/env python3
"""
generate_paper_figures.py
Gera todas as figuras para o paper:
  1. low_label_curves.png   — curvas macro-F1 vs label fraction por método
  2. stability_boxplot.png  — boxplot de macro-F1 a 10% (mostra instabilidade PPT-classic)
  3. finetune_curves.png    — curvas val_macro_f1 por época, por método (4 subplots)
  4. pretrain_loss.png      — curva train_loss do pré-treino (physio_ppt × ppt_classic)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "physio_ppt" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_DIR = ROOT / "physio_ppt" / "outputs" / "finetune"
PRETRAIN_DIR = ROOT / "physio_ppt" / "outputs" / "pretrain"
TABLES_DIR = ROOT / "physio_ppt" / "outputs" / "tables"

METHOD_LABELS = {
    "supervised_strong": "Supervised (strong)",
    "ppt_classic":       "PPT-classic",
    "wavepuzzle":        "ECGWavePuzzle",
    "physio_ppt":        "Physio-PPT (ours)",
}
METHOD_COLORS = {
    "supervised_strong": "#555555",
    "ppt_classic":       "#E07B39",
    "wavepuzzle":        "#4878CF",
    "physio_ppt":        "#2CA02C",
}
METHOD_ORDER = ["supervised_strong", "ppt_classic", "wavepuzzle", "physio_ppt"]
SMOKE = {"smoke_crop500", "smoke_test"}

# ── helper ────────────────────────────────────────────────────────────────────
def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[~df["method"].isin(SMOKE)].copy()


def load_all_test_metrics() -> pd.DataFrame:
    rows = []
    for d in FINETUNE_DIR.iterdir():
        f = d / "test_metrics.csv"
        if not f.exists():
            continue
        row = pd.read_csv(f)
        name = d.name
        if any(s in name for s in SMOKE):
            continue
        method = name.split("_frac")[0] if "_frac" in name else "unknown"
        row["method"] = method
        rows.append(row)
    df = pd.concat(rows, ignore_index=True)
    return df[~df["method"].isin(SMOKE)]


def load_events(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("event") == "epoch_end":
                rows.append(obj)
    return pd.DataFrame(rows)


# ── 1. Low-label curves ───────────────────────────────────────────────────────
def fig_low_label_curves():
    df = load_summary(TABLES_DIR / "low_label_summary.csv")
    ssl_methods = [m for m in METHOD_ORDER if m != "supervised_strong"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    metrics = [
        ("macro_f1_mean",   "macro_f1_std",   "Macro-F1"),
        ("macro_auroc_mean", None,             "Macro-AUROC"),
        ("macro_auprc_mean", None,             "Macro-AUPRC"),
    ]
    for ax, (m_mean, m_std, ylabel) in zip(axes, metrics):
        for method in ssl_methods:
            sub = df[df["method"] == method].sort_values("label_fraction")
            if sub.empty:
                continue
            x = sub["label_fraction"].values * 100  # percent
            y = sub[m_mean].values
            lbl = METHOD_LABELS.get(method, method)
            col = METHOD_COLORS.get(method, "gray")
            ax.plot(x, y, marker="o", label=lbl, color=col, linestyle="-", linewidth=1.8)
            if m_std and m_std in sub.columns:
                s = sub[m_std].fillna(0).values
                ax.fill_between(x, y - s, y + s, alpha=0.12, color=col)
        ax.set_xlabel("Labeled data (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([1, 5, 10])
        ax.set_xticklabels(["1%", "5%", "10%"])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle("Low-label performance — PTB-XL (5 seeds, mean ± std)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "low_label_curves.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ── 2. Stability boxplot ──────────────────────────────────────────────────────
def fig_stability_boxplot():
    ssl_methods = [m for m in METHOD_ORDER if m != "supervised_strong"]
    df = load_all_test_metrics()
    df10 = df[df["label_fraction"] == 0.1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (col, ylabel) in zip(axes, [("test_macro_f1", "Macro-F1"), ("test_macro_auprc", "Macro-AUPRC")]):
        data = [df10[df10["method"] == m][col].dropna().values for m in ssl_methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, method in zip(bp["boxes"], ssl_methods):
            patch.set_facecolor(METHOD_COLORS.get(method, "gray"))
            patch.set_alpha(0.7)
        # overlay individual points
        for i, (method, vals) in enumerate(zip(ssl_methods, data), 1):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=METHOD_COLORS.get(method, "gray"), s=30, zorder=5, alpha=0.8)
        ax.set_xticks(range(1, len(ssl_methods) + 1))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in ssl_methods],
                            rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-seed stability at 10% labels — PTB-XL", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "stability_boxplot.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ── 3. Finetune training curves (val macro-F1 por época) ─────────────────────
def fig_finetune_curves():
    # coletar events.jsonl de todos os runs de frac=0.10
    ssl_methods = [m for m in METHOD_ORDER if m != "supervised_strong"]
    method_epochs: dict[str, list[pd.DataFrame]] = {m: [] for m in ssl_methods}
    for d in sorted(FINETUNE_DIR.iterdir()):
        jsonl = d / "events.jsonl"
        if not jsonl.exists():
            continue
        name = d.name
        if any(s in name for s in SMOKE):
            continue
        if "frac10" not in name and "_f0.10_" not in name:
            continue
        method = name.split("_frac")[0] if "_frac" in name else "unknown"
        if method not in method_epochs:
            continue
        ev = load_events(jsonl)
        if not ev.empty and "val_macro_f1" in ev.columns:
            method_epochs[method].append(ev)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)
    axes_flat = axes.flatten()
    for ax, method in zip(axes_flat, ssl_methods):
        runs = method_epochs[method]
        col = METHOD_COLORS.get(method, "gray")
        lbl = METHOD_LABELS.get(method, method)
        for ev in runs:
            ax.plot(ev["epoch"], ev["val_macro_f1"],
                    color=col, alpha=0.45, linewidth=1.2)
        if runs:
            # mean across runs (interpolate to common epoch grid)
            max_ep = max(len(ev) for ev in runs)
            common = np.arange(1, max_ep + 1)
            vals = []
            for ev in runs:
                ep = ev["epoch"].values
                y  = ev["val_macro_f1"].values
                interp = np.interp(common, ep, y, right=np.nan)
                vals.append(interp)
            mean_y = np.nanmean(vals, axis=0)
            valid = ~np.isnan(mean_y)
            ax.plot(common[valid], mean_y[valid], color=col, linewidth=2.5,
                    label="mean", zorder=5)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Val Macro-F1", fontsize=9)
        ax.grid(alpha=0.25)
        n_runs = len(runs)
        ax.text(0.97, 0.05, f"n={n_runs} runs", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray")
    fig.suptitle("Fine-tuning curves — PTB-XL 10% labels\n(thin=individual runs, thick=mean)",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "finetune_curves.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ── 4. Pretrain loss curves ───────────────────────────────────────────────────
def fig_pretrain_loss():
    # só physio_ppt e ppt_classic têm pretrain
    pretrain_methods = {
        "physio_ppt": [],
        "ppt_classic": [],
    }
    for d in sorted(PRETRAIN_DIR.iterdir()):
        jsonl = d / "events.jsonl"
        if not jsonl.exists():
            continue
        name = d.name
        for m in pretrain_methods:
            if name.startswith(m):
                ev = load_events(jsonl)
                if not ev.empty and "train_loss" in ev.columns:
                    pretrain_methods[m].append(ev)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, runs in pretrain_methods.items():
        col = METHOD_COLORS.get(method, "gray")
        lbl = METHOD_LABELS.get(method, method)
        for ev in runs:
            ax.plot(ev["epoch"], ev["train_loss"], color=col, alpha=0.35, linewidth=1.0)
        if runs:
            max_ep = max(len(ev) for ev in runs)
            common = np.arange(1, max_ep + 1)
            vals = []
            for ev in runs:
                ep = ev["epoch"].values
                y  = ev["train_loss"].values
                vals.append(np.interp(common, ep, y, right=np.nan))
            mean_y = np.nanmean(vals, axis=0)
            valid = ~np.isnan(mean_y)
            ax.plot(common[valid], mean_y[valid], color=col, linewidth=2.5, label=lbl, zorder=5)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Train loss (SSL)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.suptitle("Pretraining loss curves — MIT-BIH\n(thin=individual seeds, thick=mean)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "pretrain_loss.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-curves", action="store_true",
                    help="Skip low_label and stability (already generated)")
    args = ap.parse_args()

    if not args.only_curves:
        print("Generating paper figures (full)...")
        fig_stability_boxplot()
    else:
        print("Skipping stability_boxplot (already exist)")

    # always regenerate these two (may have changed method filters)
    fig_low_label_curves()
    fig_finetune_curves()
    fig_pretrain_loss()
    print("Done. Figures in:", FIG_DIR)
