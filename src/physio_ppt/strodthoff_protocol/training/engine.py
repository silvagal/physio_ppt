"""Train/eval engine for record-level PTB-XL experiments."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.windowing import extract_windows, sample_random_windows
from ..evaluation.metrics import compute_multilabel_metrics, compute_per_class_report
from ..evaluation.record_aggregation import aggregate_logits
from ..utils.io import ensure_dir, save_json
from ..utils.logging import JsonlLogger, build_logger


@dataclass
class TrainArtifacts:
    run_dir: str
    best_checkpoint: str
    last_checkpoint: str
    epoch_metrics_csv: str
    test_metrics_json: str
    test_predictions_npz: str
    val_predictions_npz: str


def predict_record_level(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    window_size_samples: int,
    overlap: float,
    aggregation: str,
    eval_window_batch_size: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    logits_all = []
    y_all = []
    ecg_ids_all = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            ecg_ids = batch["ecg_id"].cpu().numpy()
            batch_logits = []
            for i in range(x.shape[0]):
                record = x[i]
                windows = extract_windows(record, window_size=window_size_samples, overlap=overlap)
                windows = windows.to(device)
                win_logits = []
                for start in range(0, windows.shape[0], int(eval_window_batch_size)):
                    chunk = windows[start : start + int(eval_window_batch_size)]
                    win_logits.append(model(chunk))
                stacked = torch.cat(win_logits, dim=0)
                record_logits = aggregate_logits(stacked, mode=aggregation)
                batch_logits.append(record_logits)
            batch_logits_t = torch.stack(batch_logits, dim=0)
            logits_all.append(batch_logits_t.cpu().numpy())
            y_all.append(y)
            ecg_ids_all.append(ecg_ids)
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "targets": np.concatenate(y_all, axis=0),
        "ecg_ids": np.concatenate(ecg_ids_all, axis=0),
    }


def _build_optimizer(model: nn.Module, train_cfg: Dict[str, object]) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _maybe_build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict[str, object],
    total_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name in {"none", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(total_epochs)))
    if name == "step":
        step_size = int(train_cfg.get("step_size", 10))
        gamma = float(train_cfg.get("step_gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {name}")


def train_record_level_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    run_dir: str | Path,
    device: str,
    seed: int,
    class_names: Sequence[str],
    train_cfg: Dict[str, object],
    protocol_cfg: Dict[str, object],
) -> TrainArtifacts:
    logger = build_logger("strodthoff_train")
    run_path = ensure_dir(run_dir)
    jsonl = JsonlLogger(run_path / "events.jsonl")
    epoch_csv = run_path / "epoch_metrics.csv"
    best_ckpt = run_path / "checkpoint_best.pt"
    last_ckpt = run_path / "checkpoint_last.pt"
    val_pred_npz = run_path / "val_predictions_record_level.npz"
    test_pred_npz = run_path / "test_predictions_record_level.npz"
    test_metrics_json = run_path / "test_metrics.json"

    model = model.to(device)
    optimizer = _build_optimizer(model, train_cfg)
    n_epochs = int(train_cfg.get("epochs", 30))
    scheduler = _maybe_build_scheduler(optimizer, train_cfg, total_epochs=n_epochs)
    criterion = nn.BCEWithLogitsLoss()

    window_size_samples = int(protocol_cfg["window_size_samples"])
    eval_overlap = float(protocol_cfg.get("eval_overlap", protocol_cfg.get("window_overlap", 0.5)))
    aggregation = str(protocol_cfg.get("aggregation", "mean"))
    eval_window_batch_size = int(protocol_cfg.get("eval_window_batch_size", 128))

    early_stop_patience = int(train_cfg.get("early_stop_patience", 10))
    model_selection_metric = str(train_cfg.get("model_selection_metric", "macro_auroc"))

    rows = []
    best_metric = -float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        generator = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
        generator.manual_seed(int(seed) * 10000 + epoch)
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            windows = sample_random_windows(x, window_size=window_size_samples, generator=generator)
            logits = model(windows)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        if scheduler is not None:
            scheduler.step()

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_pred = predict_record_level(
            model=model,
            loader=val_loader,
            device=device,
            window_size_samples=window_size_samples,
            overlap=eval_overlap,
            aggregation=aggregation,
            eval_window_batch_size=eval_window_batch_size,
        )
        val_metrics = compute_multilabel_metrics(val_pred["logits"], val_pred["targets"])
        metric_value = float(val_metrics.get(model_selection_metric, float("nan")))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        rows.append(row)
        jsonl.log("epoch_end", row)
        logger.info(
            "epoch=%d train_loss=%.4f val_macro_auroc=%.4f val_macro_f1=%.4f",
            epoch,
            train_loss,
            float(val_metrics.get("macro_auroc", float("nan"))),
            float(val_metrics.get("macro_f1", float("nan"))),
        )

        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": int(epoch),
                    "best_val_metric": float(best_metric),
                    "model_selection_metric": model_selection_metric,
                },
                best_ckpt,
            )
            np.savez_compressed(
                val_pred_npz,
                logits=val_pred["logits"],
                targets=val_pred["targets"],
                ecg_ids=val_pred["ecg_ids"],
            )
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            logger.info("early stopping at epoch=%d", epoch)
            break

    with epoch_csv.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    torch.save({"model": model.state_dict(), "epoch": len(rows)}, last_ckpt)

    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model"])
    test_pred = predict_record_level(
        model=model,
        loader=test_loader,
        device=device,
        window_size_samples=window_size_samples,
        overlap=eval_overlap,
        aggregation=aggregation,
        eval_window_batch_size=eval_window_batch_size,
    )
    np.savez_compressed(
        test_pred_npz,
        logits=test_pred["logits"],
        targets=test_pred["targets"],
        ecg_ids=test_pred["ecg_ids"],
    )
    test_metrics = compute_multilabel_metrics(test_pred["logits"], test_pred["targets"])
    test_report = compute_per_class_report(test_pred["logits"], test_pred["targets"], class_names=class_names)
    payload = {
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_metric),
        "model_selection_metric": model_selection_metric,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "test_per_class": test_report,
    }
    save_json(test_metrics_json, payload)
    jsonl.log("test_end", payload)
    jsonl.close()

    return TrainArtifacts(
        run_dir=str(run_path),
        best_checkpoint=str(best_ckpt),
        last_checkpoint=str(last_ckpt),
        epoch_metrics_csv=str(epoch_csv),
        test_metrics_json=str(test_metrics_json),
        test_predictions_npz=str(test_pred_npz),
        val_predictions_npz=str(val_pred_npz),
    )

