#!/usr/bin/env python3
"""Large-scale SSL pretraining on CODE-15 (PPT-style)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.training.pretrain_code15 import run_code15_pretrain
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config
from physio_ppt.strodthoff_protocol.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser("Pretrain SSL on CODE-15 with scalable loading")
    parser.add_argument("--config", required=True, help="Pretrain YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Runtime seed override")
    parser.add_argument("--device", type=str, default=None, help="Runtime device override")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value")
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    seed = int(args.seed if args.seed is not None else cfg.get("runtime", {}).get("seed", 42))
    device = str(args.device if args.device is not None else cfg.get("runtime", {}).get("device", "cpu"))
    deterministic = bool(cfg.get("runtime", {}).get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)

    result = run_code15_pretrain(cfg, seed=seed, device=device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

