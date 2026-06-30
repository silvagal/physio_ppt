#!/usr/bin/env python3
"""Build lazy index CSV for CODE-15 HDF5 shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.data.code15_pretrain import build_code15_hdf5_index
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config


def main() -> None:
    parser = argparse.ArgumentParser("Build CODE-15 HDF5 index for scalable pretraining")
    parser.add_argument("--config", required=True, help="Pretrain YAML config")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value")
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    dcfg = cfg["dataset"]
    stats = build_code15_hdf5_index(
        raw_root=dcfg["raw_root"],
        index_csv=dcfg["index_csv"],
        split_seed=int(dcfg.get("split_seed", 42)),
        max_records=args.max_records,
    )
    print(json.dumps({"index_csv": dcfg["index_csv"], **stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

