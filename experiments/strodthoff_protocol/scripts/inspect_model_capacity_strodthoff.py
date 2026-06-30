#!/usr/bin/env python3
"""Inspect parameter counts for XResNet and Transformer baselines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.models.factory import build_model, count_trainable_parameters
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config


def main() -> None:
    parser = argparse.ArgumentParser("Inspect model parameter counts")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    in_channels = len(cfg["dataset"].get("lead_indices", list(range(12))))
    num_classes = len(cfg["dataset"].get("class_names", ["NORM", "MI", "STTC", "CD", "HYP"]))
    model = build_model(cfg["model"], in_channels=in_channels, num_classes=num_classes)
    payload = {
        "model_name": str(cfg["model"]["name"]),
        "trainable_params": int(count_trainable_parameters(model)),
        "model_config": cfg["model"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
