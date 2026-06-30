"""CODE-15 large-scale SSL pretraining (PPT-style)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from physio_ppt.experiments.train_pretrain import build_backbone
from physio_ppt.models.heads import PPTHead
from physio_ppt.models.losses import info_nce_loss
from physio_ppt.ssl.ppt_views import permute_global_patches, strong_view_ppt, weak_view

from ..data.code15_pretrain import build_code15_hdf5_index, build_code15_pretrain_loaders
from ..utils.config import save_resolved_config
from ..utils.io import ensure_dir, save_json
from ..utils.logging import JsonlLogger, build_logger


def _sample_random_crop(batch_x: torch.Tensor, crop_size: int, generator: torch.Generator) -> torch.Tensor:
    if crop_size <= 0:
        return batch_x
    b, c, t = batch_x.shape
    if t <= crop_size:
        return batch_x
    starts = torch.randint(0, t - crop_size + 1, (b,), generator=generator, device=batch_x.device)
    out = batch_x.new_empty((b, c, crop_size))
    for i in range(b):
        s = int(starts[i].item())
        out[i] = batch_x[i, :, s : s + crop_size]
    return out


def _ppt_step(
    backbone: nn.Module,
    head: PPTHead,
    x: torch.Tensor,
    *,
    patch_len_perm: int,
    channel_perm_prob: float,
    lambda_consistency: float,
    lambda_contrast: float,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    v1 = weak_view(x)
    v2 = strong_view_ppt(x, patch_len=patch_len_perm, channel_perm_prob=channel_perm_prob)

    feat1 = backbone(v1)["pooled"]
    feat2 = backbone(v2)["pooled"]
    out1 = head(feat1)
    out2 = head(feat2)
    loss_ctr = info_nce_loss(out1["projection"], out2["projection"], temperature=temperature)

    neg = permute_global_patches(x, patch_len=patch_len_perm)
    feat_neg = backbone(neg)["pooled"]
    out_neg = head(feat_neg)
    logits = torch.cat([out1["consistency_logits"], out_neg["consistency_logits"]], dim=0)
    labels = torch.cat([torch.ones_like(out1["consistency_logits"]), torch.zeros_like(out_neg["consistency_logits"])], dim=0)
    loss_cons = F.binary_cross_entropy_with_logits(logits, labels)

    loss = lambda_consistency * loss_cons + lambda_contrast * loss_ctr
    acc_cons = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean()
    return {
        "loss": loss,
        "loss_consistency": loss_cons.detach(),
        "loss_contrast": loss_ctr.detach(),
        "acc_consistency": acc_cons.detach(),
    }


def _ensure_index(dataset_cfg: Dict[str, Any]) -> None:
    fmt = str(dataset_cfg.get("format", "hdf5_index")).lower()
    if fmt != "hdf5_index":
        return
    index_csv = Path(str(dataset_cfg["index_csv"]))
    if index_csv.exists():
        return
    raw_root = dataset_cfg.get("raw_root")
    if raw_root is None:
        raise ValueError("dataset.raw_root is required when format=hdf5_index and index_csv is missing")
    stats = build_code15_hdf5_index(
        raw_root=raw_root,
        index_csv=index_csv,
        split_seed=int(dataset_cfg.get("split_seed", 42)),
        max_records=dataset_cfg.get("max_index_records"),
    )
    meta_path = index_csv.with_suffix(".meta.json")
    save_json(meta_path, stats)


def run_code15_pretrain(cfg: Dict[str, Any], *, seed: int, device: str) -> Dict[str, Any]:
    logger = build_logger("code15_pretrain")
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    ssl_cfg = cfg.get("ssl", {})
    outputs_cfg = cfg["outputs"]

    _ensure_index(dataset_cfg)
    loaders = build_code15_pretrain_loaders(dataset_cfg, train_cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    backbone = build_backbone(model_cfg["backbone"]).to(device)
    feat_dim = int(getattr(backbone, "out_dim"))
    head = PPTHead(
        in_dim=feat_dim,
        proj_dim=int(model_cfg.get("proj_dim", 128)),
        hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
    ).to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1.5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    run_name = str(outputs_cfg.get("run_name", "pretrain_code15"))
    run_dir = ensure_dir(Path(outputs_cfg["root"]) / "pretrain" / run_name / f"seed_{seed}")
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)
    jsonl = JsonlLogger(run_dir / "events.jsonl")
    metrics_csv = run_dir / "metrics.csv"
    best_ckpt = run_dir / "best_pretrain.pt"
    latest_ckpt = run_dir / "last_pretrain.pt"

    n_epochs = int(train_cfg.get("epochs", 150))
    crop_size = int(train_cfg.get("crop_size_samples", 0))
    best_val = float("inf")
    rows = []

    for epoch in range(1, n_epochs + 1):
        backbone.train()
        head.train()
        g = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
        g.manual_seed(seed * 10000 + epoch)
        train_logs = []
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            if crop_size > 0:
                x = _sample_random_crop(x, crop_size=crop_size, generator=g)
            optimizer.zero_grad(set_to_none=True)
            m = _ppt_step(
                backbone=backbone,
                head=head,
                x=x,
                patch_len_perm=int(ssl_cfg.get("patch_len_perm", 40)),
                channel_perm_prob=float(ssl_cfg.get("channel_perm_prob", 0.3)),
                lambda_consistency=float(ssl_cfg.get("lambda_consistency", 1.0)),
                lambda_contrast=float(ssl_cfg.get("lambda_contrast", 1.0)),
                temperature=float(ssl_cfg.get("temperature", 0.2)),
            )
            loss = m["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_logs.append({k: float(v.detach().cpu().item()) for k, v in m.items()})
        if not train_logs:
            raise RuntimeError("Empty train loader in CODE-15 pretraining")

        backbone.eval()
        head.eval()
        val_logs = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                if crop_size > 0:
                    x = _sample_random_crop(x, crop_size=crop_size, generator=g)
                m = _ppt_step(
                    backbone=backbone,
                    head=head,
                    x=x,
                    patch_len_perm=int(ssl_cfg.get("patch_len_perm", 40)),
                    channel_perm_prob=float(ssl_cfg.get("channel_perm_prob", 0.3)),
                    lambda_consistency=float(ssl_cfg.get("lambda_consistency", 1.0)),
                    lambda_contrast=float(ssl_cfg.get("lambda_contrast", 1.0)),
                    temperature=float(ssl_cfg.get("temperature", 0.2)),
                )
                val_logs.append({k: float(v.detach().cpu().item()) for k, v in m.items()})
        if not val_logs:
            raise RuntimeError("Empty val loader in CODE-15 pretraining")

        train_mean = {f"train_{k}": float(np.mean([r[k] for r in train_logs])) for k in train_logs[0].keys()}
        val_mean = {f"val_{k}": float(np.mean([r[k] for r in val_logs])) for k in val_logs[0].keys()}
        row = {"epoch": epoch, **train_mean, **val_mean}
        rows.append(row)
        jsonl.log("epoch_end", row)
        val_loss = float(val_mean["val_loss"])
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f",
            epoch,
            float(train_mean["train_loss"]),
            val_loss,
        )

        ckpt = {
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "config": cfg,
            "seed": seed,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(ckpt, latest_ckpt)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_ckpt)

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt),
        "latest_checkpoint": str(latest_ckpt),
        "metrics_csv": str(metrics_csv),
        "best_val_loss": float(best_val),
        "seed": int(seed),
    }
    save_json(run_dir / "pretrain_summary.json", summary)
    jsonl.log("pretrain_end", summary)
    jsonl.close()
    return summary
