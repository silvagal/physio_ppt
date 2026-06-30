"""Fine-tuning loops for supervised evaluation and low-label regimes."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from ..data.datamodule import build_finetune_loaders
from ..models.heads import SupervisedHead
from ..models.losses import supervised_loss
from ..utils.config import config_hash
from ..utils.metrics import classification_metrics
from ..utils.io import ensure_dir, save_yaml
from ..utils.logger import JsonlLogger, build_logger
from .train_pretrain import build_backbone


class ClassifierModel(nn.Module):
    """Backbone + supervised head wrapper."""

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = SupervisedHead(in_dim=int(getattr(backbone, "out_dim")), num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)["pooled"]
        return self.head(feat)


def _collect_outputs(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> Dict[str, np.ndarray]:
    """Run inference and collect logits/targets/ids."""
    model.eval()
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    exam_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"]
            out = model(x).detach().cpu().numpy()
            logits_list.append(out)
            y_list.append(y.detach().cpu().numpy())

            if "ecg_id" in batch:
                v = batch["ecg_id"]
                if isinstance(v, list):
                    exam_ids.extend([str(z) for z in v])
                else:
                    exam_ids.extend([str(z) for z in v])
            else:
                exam_ids.extend([f"sample_{len(exam_ids)+i}" for i in range(out.shape[0])])

    logits = np.concatenate(logits_list, axis=0)
    targets = np.concatenate(y_list, axis=0)
    return {"logits": logits, "targets": targets, "exam_ids": np.asarray(exam_ids)}


def run_finetune(config: Dict[str, object], seed: int, device: str) -> Dict[str, object]:
    """Fine-tune a supervised classifier with optional pretraining init."""
    logger = build_logger()
    cfg = copy.deepcopy(config)

    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    task_type = str(train_cfg.get("task_type", "multilabel"))
    num_classes = int(train_cfg.get("num_classes", 5))

    run_hash = config_hash(cfg)
    frac = float(train_cfg.get("label_fraction", 1.0))
    run_name = f"{exp_cfg['name']}_finetune_s{seed}_f{frac:.2f}_{run_hash}"

    out_root = ensure_dir(str(cfg["paths"]["output_root"]))
    run_dir = ensure_dir(out_root / "finetune" / run_name)

    save_yaml(run_dir / "config_used.yaml", cfg)
    jsonl = JsonlLogger(run_dir / "events.jsonl")

    loaders = build_finetune_loaders(cfg, seed=seed)

    backbone = build_backbone(model_cfg["backbone"])
    model = ClassifierModel(backbone=backbone, num_classes=num_classes).to(device)

    pre_ckpt = train_cfg.get("pretrained_checkpoint")
    if pre_ckpt:
        ckpt_path = Path(str(pre_ckpt))
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = model.backbone.load_state_dict(ckpt["backbone"], strict=False)
            logger.info("loaded backbone checkpoint=%s missing=%d unexpected=%d", ckpt_path, len(missing), len(unexpected))

    if bool(train_cfg.get("freeze_backbone", False)):
        for p in model.backbone.parameters():
            p.requires_grad = False

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    n_epochs = int(train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("early_stop_patience", 5))
    use_focal = bool(train_cfg.get("use_focal", False))

    best_val_f1 = -np.inf
    best_epoch = 0
    epochs_no_improve = 0
    rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses: List[float] = []

        for batch in tqdm(loaders.train, desc=f"Finetune epoch {epoch}/{n_epochs}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = supervised_loss(logits, y, task_type=task_type, use_focal=use_focal)
            loss.backward()
            optim.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_outputs = _collect_outputs(model, loaders.val, device=device)
        val_metrics = classification_metrics(
            logits=val_outputs["logits"],
            y_true=val_outputs["targets"],
            task_type=task_type,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        rows.append(row)
        jsonl.log("epoch_end", row)

        val_f1 = float(val_metrics.get("macro_f1", float("nan")))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "seed": seed,
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_f1,
                },
                run_dir / "best_finetune.pt",
            )
        else:
            epochs_no_improve += 1

        logger.info(
            "finetune epoch=%d train_loss=%.4f val_macro_f1=%.4f",
            epoch,
            train_loss,
            val_f1,
        )

        if epochs_no_improve >= patience:
            logger.info("early stopping at epoch %d", epoch)
            break

    ckpt = torch.load(run_dir / "best_finetune.pt", map_location=device)
    model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]

    test_outputs = _collect_outputs(model, loaders.test, device=device)
    test_metrics = classification_metrics(
        logits=test_outputs["logits"],
        y_true=test_outputs["targets"],
        task_type=task_type,
    )

    np.savez_compressed(
        run_dir / "test_predictions.npz",
        logits=test_outputs["logits"],
        targets=test_outputs["targets"],
        exam_ids=test_outputs["exam_ids"],
    )

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(run_dir / "metrics.csv", index=False)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        "label_fraction": frac,
        "seed": seed,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "test_metrics.csv", index=False)
    jsonl.log("test_end", summary)
    jsonl.close()

    return {
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best_finetune.pt"),
        "metrics_csv": str(run_dir / "test_metrics.csv"),
        "predictions": str(run_dir / "test_predictions.npz"),
        **summary,
    }


__all__ = ["run_finetune"]
