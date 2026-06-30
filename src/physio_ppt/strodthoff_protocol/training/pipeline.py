"""End-to-end pipeline helpers for Strodthoff-compatible runs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ..constants import CLASS_NAMES
from ..data.low_label_sampling import generate_low_label_split
from ..data.ptbxl_strodthoff import build_loader, prepare_ptbxl_strodthoff
from ..models.factory import build_model, count_trainable_parameters, load_pretrained_backbone
from ..utils.config import save_resolved_config
from ..utils.io import ensure_dir, load_json, save_json
from ..utils.logging import build_logger
from .engine import TrainArtifacts, train_record_level_model


def _fraction_tag(fraction: float) -> str:
    pct = int(round(float(fraction) * 100.0))
    return f"{pct:02d}pct"


def ensure_prepared_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    processed_root = Path(str(dataset_cfg["processed_root"]))
    metadata_path = processed_root / "metadata.json"
    if metadata_path.exists():
        return load_json(metadata_path)
    if not bool(dataset_cfg.get("prepare_if_missing", True)):
        raise FileNotFoundError(
            f"{metadata_path} not found and dataset.prepare_if_missing=false. "
            "Run prepare_ptbxl_strodthoff.py first."
        )
    return prepare_ptbxl_strodthoff(
        raw_root=dataset_cfg["raw_root"],
        processed_root=processed_root,
        fs=int(dataset_cfg.get("fs", 100)),
        normalize=str(dataset_cfg.get("normalize", "zscore")),
        lead_indices=dataset_cfg.get("lead_indices"),
        class_names=dataset_cfg.get("class_names", CLASS_NAMES),
    )


def ensure_low_label_file(
    cfg: Dict[str, Any],
    *,
    split_seed: int,
    fraction: float,
) -> str | None:
    if float(fraction) >= 1.0:
        return None
    ds_root = Path(str(cfg["dataset"]["processed_root"]))
    train_npz = np.load(ds_root / "train" / "records.npz", allow_pickle=True)
    train_ecg_ids = train_npz["ecg_ids"].astype(np.int64)
    split_root = Path(str(cfg["dataset"]["low_label_splits_root"]))
    split_path = split_root / f"seed_{split_seed}" / f"fraction_{_fraction_tag(fraction)}.json"
    if split_path.exists():
        return str(split_path)
    result = generate_low_label_split(
        train_ecg_ids=train_ecg_ids,
        fraction=float(fraction),
        seed=int(split_seed),
        out_dir=split_root,
        method=str(cfg.get("low_label", {}).get("split_method", "record_random")),
    )
    return str(result["split_file"])


def _load_selected_ids(split_file: str | None) -> Sequence[int] | None:
    if not split_file:
        return None
    payload = load_json(split_file)
    selected = payload.get("selected_ecg_ids", [])
    return [int(x) for x in selected]


def _build_run_dir(cfg: Dict[str, Any], *, seed: int, fraction: float) -> Path:
    out_root = ensure_dir(str(cfg["outputs"]["root"]))
    run_name = str(cfg["outputs"].get("run_name", cfg.get("experiment", {}).get("name", "run")))
    return ensure_dir(out_root / "runs" / run_name / f"seed_{seed}" / f"fraction_{_fraction_tag(fraction)}")


def run_training_pipeline(
    cfg: Dict[str, Any],
    *,
    seed: int,
    device: str,
    use_pretrained: bool,
) -> Dict[str, Any]:
    logger = build_logger("strodthoff_pipeline")
    ensure_prepared_dataset(cfg)
    fraction = float(cfg.get("low_label", {}).get("fraction", 1.0))
    split_seed = int(cfg.get("low_label", {}).get("split_seed", seed))
    split_file = ensure_low_label_file(cfg, split_seed=split_seed, fraction=fraction)
    selected_ids = _load_selected_ids(split_file)

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    protocol_cfg = dict(cfg.get("protocol", {}))
    fs = int(dataset_cfg.get("fs", 100))
    protocol_cfg["window_size_samples"] = int(round(float(protocol_cfg.get("window_size_seconds", 2.5)) * fs))

    ds_root = dataset_cfg["processed_root"]
    train_loader = build_loader(
        root=ds_root,
        split="train",
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=True,
        selected_ecg_ids=selected_ids,
    )
    val_loader = build_loader(
        root=ds_root,
        split="val",
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=False,
        selected_ecg_ids=None,
    )
    test_loader = build_loader(
        root=ds_root,
        split="test",
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=False,
        selected_ecg_ids=None,
    )

    in_channels = len(dataset_cfg.get("lead_indices", list(range(12))))
    num_classes = len(dataset_cfg.get("class_names", CLASS_NAMES))
    model = build_model(cfg["model"], in_channels=in_channels, num_classes=num_classes)
    model_params = count_trainable_parameters(model)
    logger.info("model=%s trainable_params=%d", cfg["model"]["name"], model_params)

    pretrain_report: Dict[str, Any] | None = None
    if use_pretrained:
        pre_cfg = cfg.get("pretrained", {})
        checkpoint = str(pre_cfg.get("checkpoint", ""))
        if not checkpoint:
            raise ValueError("pretrained.checkpoint must be set for SSL fine-tune runs")
        pretrain_report = load_pretrained_backbone(
            model,
            checkpoint_path=checkpoint,
            source=str(pre_cfg.get("source", "physio_ppt_pretrain")),
        )
        logger.info("pretrained_load=%s", pretrain_report)

    run_dir = _build_run_dir(cfg, seed=seed, fraction=fraction)
    resolved_cfg = dict(cfg)
    resolved_cfg.setdefault("runtime", {})
    resolved_cfg["runtime"]["seed"] = int(seed)
    resolved_cfg["runtime"]["device"] = str(device)
    resolved_cfg["runtime"]["label_fraction"] = float(fraction)
    resolved_cfg["runtime"]["low_label_split_file"] = split_file
    resolved_cfg["runtime"]["trainable_params"] = int(model_params)
    if pretrain_report is not None:
        resolved_cfg["runtime"]["pretrained_report"] = pretrain_report
    save_resolved_config(run_dir / "config_resolved.yaml", resolved_cfg)

    artifacts: TrainArtifacts = train_record_level_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run_dir=run_dir,
        device=device,
        seed=seed,
        class_names=dataset_cfg.get("class_names", CLASS_NAMES),
        train_cfg=train_cfg,
        protocol_cfg=protocol_cfg,
    )

    result = {
        "experiment_name": str(cfg.get("experiment", {}).get("name", "unknown")),
        "seed": int(seed),
        "fraction": float(fraction),
        "device": str(device),
        "split_seed": int(split_seed),
        "split_file": split_file,
        "model_name": str(cfg["model"]["name"]),
        "trainable_params": int(model_params),
        "use_pretrained": bool(use_pretrained),
        "pretrained_report": pretrain_report,
        "run_dir": artifacts.run_dir,
        "best_checkpoint": artifacts.best_checkpoint,
        "last_checkpoint": artifacts.last_checkpoint,
        "epoch_metrics_csv": artifacts.epoch_metrics_csv,
        "test_metrics_json": artifacts.test_metrics_json,
        "test_predictions_npz": artifacts.test_predictions_npz,
        "val_predictions_npz": artifacts.val_predictions_npz,
    }
    save_json(Path(artifacts.run_dir) / "run_summary.json", result)

    ledger = ensure_dir(Path(cfg["outputs"]["root"]) / "tables") / "runs_ledger.csv"
    _append_ledger(ledger, result)
    return result


def _append_ledger(path: Path, row: Dict[str, Any]) -> None:
    flat = {
        "experiment_name": row["experiment_name"],
        "seed": row["seed"],
        "fraction": row["fraction"],
        "model_name": row["model_name"],
        "use_pretrained": row["use_pretrained"],
        "trainable_params": row["trainable_params"],
        "run_dir": row["run_dir"],
        "test_metrics_json": row["test_metrics_json"],
        "split_file": row["split_file"] or "",
        "best_checkpoint": row["best_checkpoint"],
    }
    fields: List[str] = list(flat.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(flat)

