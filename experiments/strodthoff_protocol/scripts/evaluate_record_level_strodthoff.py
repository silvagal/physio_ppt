#!/usr/bin/env python3
"""Evaluate a trained checkpoint with record-level sliding-window inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "physio_ppt" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physio_ppt.strodthoff_protocol.constants import CLASS_NAMES
from physio_ppt.strodthoff_protocol.data.ptbxl_strodthoff import build_loader
from physio_ppt.strodthoff_protocol.evaluation.metrics import compute_multilabel_metrics, compute_per_class_report
from physio_ppt.strodthoff_protocol.models.factory import build_model
from physio_ppt.strodthoff_protocol.training.engine import predict_record_level
from physio_ppt.strodthoff_protocol.training.pipeline import ensure_prepared_dataset
from physio_ppt.strodthoff_protocol.utils.config import load_protocol_config
from physio_ppt.strodthoff_protocol.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser("Record-level evaluation for Strodthoff protocol")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt or checkpoint_last.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="", help="Optional output directory")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value")
    args = parser.parse_args()

    cfg = load_protocol_config(args.config, overrides=args.override)
    ensure_prepared_dataset(cfg)
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    proto_cfg = cfg.get("protocol", {})

    in_channels = len(ds_cfg.get("lead_indices", list(range(12))))
    class_names = ds_cfg.get("class_names", CLASS_NAMES)
    model = build_model(cfg["model"], in_channels=in_channels, num_classes=len(class_names))
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(args.device)

    loader = build_loader(
        root=ds_cfg["processed_root"],
        split=args.split,
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=False,
        selected_ecg_ids=None,
    )
    fs = int(ds_cfg.get("fs", 100))
    window_size_samples = int(round(float(proto_cfg.get("window_size_seconds", 2.5)) * fs))
    preds = predict_record_level(
        model=model,
        loader=loader,
        device=args.device,
        window_size_samples=window_size_samples,
        overlap=float(proto_cfg.get("eval_overlap", proto_cfg.get("window_overlap", 0.5))),
        aggregation=str(proto_cfg.get("aggregation", "mean")),
        eval_window_batch_size=int(proto_cfg.get("eval_window_batch_size", 256)),
    )
    metrics = compute_multilabel_metrics(preds["logits"], preds["targets"])
    per_class = compute_per_class_report(preds["logits"], preds["targets"], class_names=class_names)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / f"eval_{args.split}"
    out_dir = ensure_dir(out_dir)
    np.savez_compressed(
        out_dir / f"predictions_{args.split}.npz",
        logits=preds["logits"],
        targets=preds["targets"],
        ecg_ids=preds["ecg_ids"],
    )
    payload = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "per_class": per_class,
        "load_state_missing_keys": [str(x) for x in missing],
        "load_state_unexpected_keys": [str(x) for x in unexpected],
    }
    save_json(out_dir / f"metrics_{args.split}.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
