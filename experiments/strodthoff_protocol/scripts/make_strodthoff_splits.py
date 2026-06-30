#!/usr/bin/env python3
"""Prepare official-fold dataset (if needed) and generate low-label split files."""
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
from physio_ppt.strodthoff_protocol.training.pipeline import ensure_prepared_dataset
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config
from physio_ppt.strodthoff_protocol.utils.io import ensure_dir


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser("Prepare official splits and create low-label subsets")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--fractions", default="0.01,0.05,0.10")
    parser.add_argument("--seeds", default="42,1337,2025")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value")
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    ensure_prepared_dataset(cfg)
    ds_cfg = cfg["dataset"]
    split_root = Path(str(ds_cfg["low_label_splits_root"]))

    train_npz = np.load(Path(str(ds_cfg["processed_root"])) / "train" / "records.npz", allow_pickle=True)
    train_ecg_ids = train_npz["ecg_ids"].astype(np.int64)

    fractions = _parse_floats(args.fractions)
    seeds = _parse_ints(args.seeds)
    rows = []
    for seed in seeds:
        for frac in fractions:
            result = generate_low_label_split(
                train_ecg_ids=train_ecg_ids,
                fraction=float(frac),
                seed=int(seed),
                out_dir=split_root,
                method=str(cfg.get("low_label", {}).get("split_method", "record_random")),
            )
            rows.append(
                {
                    "seed": int(seed),
                    "fraction": float(frac),
                    "n_full": int(result["num_train_records_full"]),
                    "n_selected": int(result["num_train_records_selected"]),
                    "split_file": str(result["split_file"]),
                }
            )

    manifest = ensure_dir(split_root) / "splits_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(f"Prepared official dataset and saved split manifest: {manifest}")


if __name__ == "__main__":
    main()

