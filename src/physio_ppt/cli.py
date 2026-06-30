"""Command-line interface for Physio-PPT experiments."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .analysis.acf_cos import aggregate_acf_cos, analyze_npz_signals, spearman_orderness_gain
from .analysis.beat_order_score import aggregate_beat_order_score, analyze_beats_npz
from .data.preprocess import prepare_all
from .experiments.run_pipeline import run_pipeline
from .experiments.train_finetune import run_finetune
from .experiments.train_pretrain import run_pretrain
from .utils.bootstrap import as_dict, paired_bootstrap
from .utils.config import apply_overrides, load_config
from .utils.io import ensure_dir, save_json
from .utils.logger import build_logger
from .utils.metrics import classification_metrics
from .utils.plot import (
    plot_ablation_bars,
    plot_low_label_curves,
    plot_orderness_scatter,
    plot_seed_delta,
)
from .utils.seed import set_global_seed


def _prepare_cfg(config_path: str, overrides: List[str], seed: int, device: str) -> Dict[str, object]:
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    cfg.setdefault("runtime", {})
    cfg["runtime"]["seed"] = seed
    cfg["runtime"]["device"] = device

    if "data" in cfg:
        cfg["data"].setdefault("seed", seed)
    return cfg


def cmd_prepare_data(cfg: Dict[str, object]) -> Dict[str, object]:
    return prepare_all(cfg["data"])


def cmd_pretrain(cfg: Dict[str, object], seed: int, device: str) -> Dict[str, object]:
    return run_pretrain(cfg, seed=seed, device=device)


def cmd_finetune(cfg: Dict[str, object], seed: int, device: str) -> Dict[str, object]:
    return run_finetune(cfg, seed=seed, device=device)


def _metric_macro_f1(y_true: np.ndarray, logits: np.ndarray, task_type: str) -> float:
    m = classification_metrics(logits=logits, y_true=y_true, task_type=task_type)
    return float(m["macro_f1"])


def cmd_eval(cfg: Dict[str, object]) -> Dict[str, object]:
    logger = build_logger()
    eval_cfg = cfg["eval"]
    kind = str(eval_cfg.get("kind", "aggregate"))

    out_dir = ensure_dir(str(cfg["paths"]["output_root"]))
    table_dir = ensure_dir(Path(out_dir) / "tables")

    if kind == "aggregate":
        glob_pat = str(eval_cfg.get("glob", str(Path(out_dir) / "finetune" / "*" / "test_metrics.csv")))
        files = sorted(Path(".").glob(glob_pat)) if not glob_pat.startswith("/") else sorted(Path("/").glob(glob_pat[1:]))
        if not files:
            files = [Path(p) for p in sorted(Path(out_dir).glob("finetune/*/test_metrics.csv"))]
        if not files:
            raise FileNotFoundError("No test_metrics.csv files found for aggregation")

        rows = []
        for f in files:
            row_df = pd.read_csv(f)
            if "method" not in row_df.columns:
                # Extract method from dir name: everything before the first '_frac'
                dir_name = f.parent.name
                row_df["method"] = dir_name.split("_frac")[0] if "_frac" in dir_name else "unknown"
            rows.append(row_df)
        df = pd.concat(rows, ignore_index=True)
        if "method" not in df.columns:
            df["method"] = str(cfg.get("experiment", {}).get("method", "unknown"))
        if "label_fraction" not in df.columns:
            df["label_fraction"] = float(cfg.get("train", {}).get("label_fraction", 1.0))

        summary = (
            df.groupby(["method", "label_fraction"], as_index=False)
            .agg(
                macro_f1_mean=("test_macro_f1", "mean"),
                macro_f1_std=("test_macro_f1", "std"),
                macro_auroc_mean=("test_macro_auroc", "mean"),
                macro_auprc_mean=("test_macro_auprc", "mean"),
                micro_f1_mean=("test_micro_f1", "mean"),
                n=("seed", "count"),
            )
            .sort_values(["method", "label_fraction"])
        )
        out = table_dir / "low_label_summary.csv"
        summary.to_csv(out, index=False)
        logger.info("saved aggregate eval to %s", out)
        return {"summary_csv": str(out)}

    if kind == "bootstrap":
        a_path = Path(str(eval_cfg["pred_a"]))
        b_path = Path(str(eval_cfg["pred_b"]))
        task_type = str(eval_cfg.get("task_type", "multilabel"))
        n_boot = int(eval_cfg.get("n_bootstrap", 10_000))
        seed = int(eval_cfg.get("seed", 42))

        a = np.load(a_path, allow_pickle=True)
        b = np.load(b_path, allow_pickle=True)
        y_true = a["targets"]
        if y_true.shape[0] != b["targets"].shape[0]:
            raise AssertionError("Prediction files have different number of samples")

        def _metric(y: np.ndarray, pred: np.ndarray) -> float:
            return _metric_macro_f1(y_true=y, logits=pred, task_type=task_type)

        res = paired_bootstrap(
            y_true=y_true,
            pred_a=a["logits"],
            pred_b=b["logits"],
            metric_fn=_metric,
            n_bootstrap=n_boot,
            seed=seed,
        )
        out_json = table_dir / "bootstrap_significance.json"
        save_json(out_json, as_dict(res))
        logger.info("saved bootstrap result to %s", out_json)
        return {"bootstrap_json": str(out_json), **as_dict(res)}

    raise ValueError(f"Unsupported eval kind: {kind}")


def cmd_analyze(cfg: Dict[str, object]) -> Dict[str, object]:
    logger = build_logger()
    analysis_cfg = cfg["analysis"]
    out_dir = ensure_dir(str(cfg["paths"]["output_root"]))
    table_dir = ensure_dir(Path(out_dir) / "tables")

    windows_npz = Path(str(analysis_cfg["windows_npz"]))
    beats_npz = Path(str(analysis_cfg["beats_npz"]))

    acf_df = analyze_npz_signals(
        npz_path=windows_npz,
        max_lag=int(analysis_cfg.get("acf_max_lag", 200)),
        chunk=int(analysis_cfg.get("acf_chunk", 40)),
    )
    bos_df = analyze_beats_npz(
        npz_path=beats_npz,
        fs=int(analysis_cfg.get("fs", 500)),
        pre_ms=int(analysis_cfg.get("beat_pre_ms", 200)),
    )

    acf_path = table_dir / "acf_cos_scores.csv"
    bos_path = table_dir / "beat_order_scores.csv"
    acf_df.to_csv(acf_path, index=False)
    bos_df.to_csv(bos_path, index=False)

    summary = {
        **aggregate_acf_cos(acf_df),
        **aggregate_beat_order_score(bos_df),
    }

    perf_csv = analysis_cfg.get("performance_csv")
    if perf_csv and Path(str(perf_csv)).exists():
        perf = pd.read_csv(str(perf_csv))
        if {"orderness", "gain"}.issubset(set(perf.columns)):
            corr, p = spearman_orderness_gain(perf["orderness"].to_numpy(), perf["gain"].to_numpy())
            summary["spearman_corr"] = corr
            summary["spearman_p"] = p

    out_json = table_dir / "orderness_summary.json"
    save_json(out_json, summary)
    logger.info("saved analysis summary to %s", out_json)

    return {
        "acf_csv": str(acf_path),
        "bos_csv": str(bos_path),
        "summary_json": str(out_json),
        **summary,
    }


def cmd_make_figures(cfg: Dict[str, object]) -> Dict[str, object]:
    fig_cfg = cfg["figures"]
    fig_dir = ensure_dir(str(cfg["paths"].get("figures_dir", "physio_ppt/figures")))

    out = {}

    low_label_csv = fig_cfg.get("low_label_csv")
    if low_label_csv and Path(str(low_label_csv)).exists():
        df = pd.read_csv(str(low_label_csv))
        p = Path(fig_dir) / "low_label_curves.png"
        plot_low_label_curves(df, metric=str(fig_cfg.get("low_label_metric", "macro_f1_mean")), out_path=p)
        out["low_label_fig"] = str(p)

    ablation_csv = fig_cfg.get("ablation_csv")
    if ablation_csv and Path(str(ablation_csv)).exists():
        df = pd.read_csv(str(ablation_csv))
        p = Path(fig_dir) / "ablation_bars.png"
        plot_ablation_bars(df, metric=str(fig_cfg.get("ablation_metric", "macro_f1")), out_path=p)
        out["ablation_fig"] = str(p)

    delta_csv = fig_cfg.get("delta_csv")
    if delta_csv and Path(str(delta_csv)).exists():
        df = pd.read_csv(str(delta_csv))
        p = Path(fig_dir) / "delta_by_seed.png"
        plot_seed_delta(df, out_path=p)
        out["delta_fig"] = str(p)

    scatter_csv = fig_cfg.get("orderness_csv")
    if scatter_csv and Path(str(scatter_csv)).exists():
        df = pd.read_csv(str(scatter_csv))
        p = Path(fig_dir) / "orderness_vs_gain.png"
        plot_orderness_scatter(df, out_path=p)
        out["orderness_fig"] = str(p)

    return out


def main() -> None:
    parser = argparse.ArgumentParser("Physio-PPT CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="Path to YAML config")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--device", type=str, default="cpu")
    common.add_argument("--override", action="append", default=[], help="Config override key=value")

    sub.add_parser("prepare_data", parents=[common])
    sub.add_parser("pretrain", parents=[common])
    sub.add_parser("finetune", parents=[common])
    sub.add_parser("eval", parents=[common])
    sub.add_parser("analyze", parents=[common])
    sub.add_parser("make_figures", parents=[common])
    sub.add_parser("run", parents=[common])

    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = _prepare_cfg(args.config, args.override, seed=args.seed, device=args.device)

    if args.command == "prepare_data":
        out = cmd_prepare_data(cfg)
    elif args.command == "pretrain":
        out = cmd_pretrain(cfg, seed=args.seed, device=args.device)
    elif args.command == "finetune":
        out = cmd_finetune(cfg, seed=args.seed, device=args.device)
    elif args.command == "eval":
        out = cmd_eval(cfg)
    elif args.command == "analyze":
        out = cmd_analyze(cfg)
    elif args.command == "make_figures":
        out = cmd_make_figures(cfg)
    elif args.command == "run":
        out = run_pipeline(cfg, seed=args.seed, device=args.device)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(pd.Series(out).to_string())


if __name__ == "__main__":
    main()
