#!/usr/bin/env python3
"""Bootstrap CI for Macro-AUROC from record-level predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.evaluation.bootstrap import bootstrap_metric_ci
from physio_ppt.strodthoff_protocol.evaluation.metrics import compute_multilabel_metrics
from physio_ppt.strodthoff_protocol.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser("Bootstrap Macro-AUROC CI")
    parser.add_argument("--predictions_npz", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    data = np.load(args.predictions_npz, allow_pickle=True)
    logits = data["logits"]
    y_true = data["targets"]

    def metric_fn(y: np.ndarray, lg: np.ndarray) -> float:
        return float(compute_multilabel_metrics(lg, y)["macro_auroc"])

    ci = bootstrap_metric_ci(
        y_true=y_true,
        logits=logits,
        metric_fn=metric_fn,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    payload = {"predictions_npz": str(args.predictions_npz), **ci}
    if args.output_json:
        save_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
