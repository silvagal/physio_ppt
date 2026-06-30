#!/usr/bin/env python3
"""Prepare PTB-XL using official folds (1-8/9/10) for Strodthoff-compatible downstream."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.data.ptbxl_strodthoff import prepare_ptbxl_strodthoff
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config
from physio_ppt.strodthoff_protocol.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser("Prepare PTB-XL official-fold dataset for Strodthoff protocol")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value")
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    dataset_cfg = cfg["dataset"]
    metadata = prepare_ptbxl_strodthoff(
        raw_root=dataset_cfg["raw_root"],
        processed_root=dataset_cfg["processed_root"],
        fs=int(dataset_cfg.get("fs", 100)),
        normalize=str(dataset_cfg.get("normalize", "zscore")),
        lead_indices=dataset_cfg.get("lead_indices"),
        class_names=dataset_cfg.get("class_names"),
    )
    out_path = Path(dataset_cfg["processed_root"]) / "prepare_summary.json"
    save_json(out_path, metadata)
    print(f"Prepared dataset: {out_path}")


if __name__ == "__main__":
    main()
