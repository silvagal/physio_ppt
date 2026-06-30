#!/usr/bin/env python3
"""Fine-tune pretrained SSL backbones under Strodthoff-compatible PTB-XL protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.training.pipeline import run_training_pipeline
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config
from physio_ppt.strodthoff_protocol.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser("SSL fine-tuning for Strodthoff-compatible downstream")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Runtime seed override")
    parser.add_argument("--device", type=str, default=None, help="Runtime device override")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value")
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    seed = int(args.seed if args.seed is not None else cfg.get("runtime", {}).get("seed", 42))
    device = str(args.device if args.device is not None else cfg.get("runtime", {}).get("device", "cpu"))
    deterministic = bool(cfg.get("runtime", {}).get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)

    result = run_training_pipeline(cfg, seed=seed, device=device, use_pretrained=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
