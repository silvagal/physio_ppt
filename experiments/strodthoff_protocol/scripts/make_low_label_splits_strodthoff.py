#!/usr/bin/env python3
"""Generate persistent low-label train subsets for Strodthoff-compatible PTB-XL."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.data.low_label_sampling import generate_low_label_split
from physio_ppt.strodthoff_protocol.data.ptbxl_strodthoff import prepare_ptbxl_strodthoff
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config
from physio_ppt.strodthoff_protocol.utils.io import ensure_dir


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser("Create low-label split files for official PTB-XL train folds")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--fractions", default="0.01,0.05,0.10", help="Comma-separated label fractions")
    parser.add_argument("--seeds", default="42,1337,2025", help="Comma-separated split seeds")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value")
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    dataset_cfg = cfg["dataset"]

    processed_root = Path(str(dataset_cfg["processed_root"]))
    if not (processed_root / "train" / "records.npz").exists():
        prepare_ptbxl_strodthoff(
            raw_root=dataset_cfg["raw_root"],
            processed_root=processed_root,
            fs=int(dataset_cfg.get("fs", 100)),
            normalize=str(dataset_cfg.get("normalize", "zscore")),
            lead_indices=dataset_cfg.get("lead_indices"),
            class_names=dataset_cfg.get("class_names"),
        )

    train_npz = np.load(processed_root / "train" / "records.npz", allow_pickle=True)
    train_ecg_ids = train_npz["ecg_ids"].astype(np.int64)

    split_root = Path(str(dataset_cfg["low_label_splits_root"]))
    fractions = _parse_floats(args.fractions)
    seeds = _parse_ints(args.seeds)
    summary_rows = []
    for seed in seeds:
        for frac in fractions:
            res = generate_low_label_split(
                train_ecg_ids=train_ecg_ids,
                fraction=float(frac),
                seed=int(seed),
                out_dir=split_root,
                method=str(cfg.get("low_label", {}).get("split_method", "record_random")),
            )
            summary_rows.append(
                {
                    "seed": int(seed),
                    "fraction": float(frac),
                    "num_train_records_full": int(res["num_train_records_full"]),
                    "num_train_records_selected": int(res["num_train_records_selected"]),
                    "split_file": str(res["split_file"]),
                    "method": str(res["method"]),
                }
            )

    summary_csv = ensure_dir(split_root) / "splits_manifest.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
    print(f"Saved low-label splits manifest: {summary_csv}")


if __name__ == "__main__":
    main()
