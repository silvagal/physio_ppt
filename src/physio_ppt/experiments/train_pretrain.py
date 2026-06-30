"""Pretraining loops for PPT, WavePuzzle, Physio-PPT and Hybrid."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ..data.datamodule import build_pretrain_loaders
from ..models.heads import PhysioPPTHead, PPTHead, WavePuzzleHead
from ..models.losses import info_nce_loss
from ..models.resnet1d import ResNet1DEncoder
from ..models.transformer import ECGTransformerEncoder
from ..ssl.hybrid_task import hybrid_step
from ..ssl.physio_ppt_task import physio_ppt_step
from ..ssl.ppt_views import permute_global_patches, strong_view_ppt, weak_view
from ..ssl.wavepuzzle_task import wavepuzzle_step
from ..utils.config import config_hash
from ..utils.io import ensure_dir, save_yaml
from ..utils.logger import JsonlLogger, build_logger


def build_backbone(model_cfg: Dict[str, object]) -> nn.Module:
    """Instantiate backbone from config dictionary."""
    name = str(model_cfg.get("name", "resnet1d")).lower()
    in_channels = int(model_cfg.get("in_channels", 12))

    if name == "resnet1d":
        return ResNet1DEncoder(
            in_channels=in_channels,
            base_channels=int(model_cfg.get("base_channels", 64)),
            d_model=int(model_cfg.get("d_model", 256)),
            blocks_per_stage=tuple(model_cfg.get("blocks_per_stage", [2, 2, 2])),
        )
    if name == "transformer":
        return ECGTransformerEncoder(
            in_channels=in_channels,
            patch_len=int(model_cfg.get("patch_len", 125)),
            stride=int(model_cfg.get("stride", 62)),
            d_model=int(model_cfg.get("d_model", 192)),
            nhead=int(model_cfg.get("nhead", 6)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pooling=str(model_cfg.get("pooling", "mean")),
            learnable_positional=bool(model_cfg.get("learnable_positional", False)),
            max_tokens=int(model_cfg.get("max_tokens", 4096)),
            use_channel_embedding=bool(model_cfg.get("use_channel_embedding", True)),
        )
    raise ValueError(f"Unsupported backbone name: {name}")


def _scalar_metrics(m: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in m.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu().item())
        else:
            out[k] = float(v)
    return out


def _ppt_step(
    backbone: nn.Module,
    head: PPTHead,
    x: torch.Tensor,
    cfg: Dict[str, object],
) -> Dict[str, torch.Tensor]:
    ssl_cfg = cfg.get("ssl", {})
    patch_len = int(ssl_cfg.get("patch_len_perm", 40))
    lambda_cons = float(ssl_cfg.get("lambda_consistency", 1.0))
    lambda_ctr = float(ssl_cfg.get("lambda_contrast", 1.0))
    temperature = float(ssl_cfg.get("temperature", 0.2))

    v1 = weak_view(x)
    v2 = strong_view_ppt(
        x,
        patch_len=patch_len,
        channel_perm_prob=float(ssl_cfg.get("channel_perm_prob", 0.3)),
    )

    feat1 = backbone(v1)["pooled"]
    feat2 = backbone(v2)["pooled"]
    out1 = head(feat1)
    out2 = head(feat2)

    loss_ctr = info_nce_loss(out1["projection"], out2["projection"], temperature=temperature)

    neg = permute_global_patches(x, patch_len=patch_len)
    feat_neg = backbone(neg)["pooled"]
    out_neg = head(feat_neg)

    logits = torch.cat([out1["consistency_logits"], out_neg["consistency_logits"]], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(out1["consistency_logits"]),
            torch.zeros_like(out_neg["consistency_logits"]),
        ],
        dim=0,
    )
    loss_cons = F.binary_cross_entropy_with_logits(logits, labels)

    total = lambda_cons * loss_cons + lambda_ctr * loss_ctr
    acc_cons = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean()

    return {
        "loss": total,
        "loss_consistency": loss_cons.detach(),
        "loss_contrast": loss_ctr.detach(),
        "acc_consistency": acc_cons.detach(),
    }


def run_pretrain(config: Dict[str, object], seed: int, device: str) -> Dict[str, object]:
    """Run SSL pretraining according to config."""
    logger = build_logger()
    cfg = copy.deepcopy(config)

    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    method = str(exp_cfg["method"])
    # Compute the hash only over fields that affect SSL pretraining.
    # Finetune-only keys (label_fractions, do_pretrain) and metadata
    # (_config_path) are excluded so that different downstream overrides
    # reuse the same pretrain checkpoint directory.
    _hash_cfg = copy.deepcopy(cfg)
    _hash_cfg.pop("_config_path", None)
    _pipeline = _hash_cfg.get("pipeline", {})
    for _finetune_key in ("label_fractions", "do_pretrain"):
        _pipeline.pop(_finetune_key, None)
    run_hash = config_hash(_hash_cfg)
    run_name = f"{exp_cfg['name']}_{method}_s{seed}_{run_hash}"

    out_root = ensure_dir(str(cfg["paths"]["output_root"]))
    run_dir = ensure_dir(out_root / "pretrain" / run_name)

    # Skip pretraining if a completed checkpoint already exists (e.g. when
    # E3 runs share the same pretrain config as E1).
    ckpt_path = run_dir / "best_pretrain.pt"
    if ckpt_path.exists():
        logger.info("pretrain checkpoint already exists, skipping training: %s", run_dir)
        metrics_csv = run_dir / "metrics.csv"
        return {
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "metrics_csv": str(metrics_csv) if metrics_csv.exists() else None,
        }

    save_yaml(run_dir / "config_used.yaml", cfg)
    jsonl = JsonlLogger(run_dir / "events.jsonl")

    loaders = build_pretrain_loaders(cfg, seed=seed)

    backbone = build_backbone(model_cfg["backbone"]).to(device)
    feat_dim = int(getattr(backbone, "out_dim"))

    head: nn.Module
    extra_head: nn.Module | None = None
    if method == "ppt_classic":
        head = PPTHead(in_dim=feat_dim, proj_dim=int(model_cfg["head"].get("proj_dim", 128))).to(device)
    elif method == "wavepuzzle":
        head = WavePuzzleHead(
            in_dim=feat_dim,
            num_permutations=int(model_cfg["head"].get("num_permutations", 6)),
            proj_dim=int(model_cfg["head"].get("proj_dim", 128)),
        ).to(device)
    elif method == "physio_ppt":
        head = PhysioPPTHead(
            in_dim=feat_dim,
            proj_dim=int(model_cfg["head"].get("proj_dim", 128)),
            enable_segment_order=bool(model_cfg["head"].get("enable_segment_order", True)),
            num_segment_orders=int(model_cfg["head"].get("num_segment_orders", 6)),
        ).to(device)
    elif method == "hybrid":
        head = PhysioPPTHead(
            in_dim=feat_dim,
            proj_dim=int(model_cfg["head"].get("proj_dim", 128)),
            enable_segment_order=bool(model_cfg["head"].get("enable_segment_order", True)),
            num_segment_orders=int(model_cfg["head"].get("num_segment_orders", 6)),
        ).to(device)
        extra_head = WavePuzzleHead(
            in_dim=feat_dim,
            num_permutations=int(model_cfg["head"].get("num_permutations", 6)),
            proj_dim=int(model_cfg["head"].get("proj_dim", 128)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported pretrain method: {method}")

    params = list(backbone.parameters()) + list(head.parameters())
    if extra_head is not None:
        params += list(extra_head.parameters())

    optim = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    n_epochs = int(train_cfg.get("epochs", 10))
    best_val = float("inf")
    metrics_rows = []

    for epoch in range(1, n_epochs + 1):
        backbone.train()
        head.train()
        if extra_head is not None:
            extra_head.train()

        train_logs = []
        pbar = tqdm(loaders.train, desc=f"Pretrain {method} epoch {epoch}/{n_epochs}")
        for batch in pbar:
            x = batch["x"].to(device)
            optim.zero_grad(set_to_none=True)

            if method == "ppt_classic":
                m = _ppt_step(backbone=backbone, head=head, x=x, cfg=cfg)
            elif method == "wavepuzzle":
                m = wavepuzzle_step(
                    backbone=backbone,
                    head=head,
                    x=x,
                    fs=int(cfg["data"].get("fs", 500)),
                    lambda_contrast=float(cfg.get("ssl", {}).get("lambda_contrast", 0.2)),
                    temperature=float(cfg.get("ssl", {}).get("temperature", 0.2)),
                )
            elif method == "physio_ppt":
                m = physio_ppt_step(
                    backbone=backbone,
                    head=head,
                    x=x,
                    fs=int(cfg["data"].get("fs", 500)),
                    pre_ms=int(cfg.get("ssl", {}).get("beat_pre_ms", 200)),
                    lambda_consistency=float(cfg.get("ssl", {}).get("lambda_consistency", 1.0)),
                    lambda_contrast=float(cfg.get("ssl", {}).get("lambda_contrast", 1.0)),
                    lambda_segment=float(cfg.get("ssl", {}).get("lambda_segment", 0.2)),
                    perturb_mode=str(cfg.get("ssl", {}).get("perturb_mode", "mixed")),
                    temperature=float(cfg.get("ssl", {}).get("temperature", 0.2)),
                    use_consistency=bool(cfg.get("ssl", {}).get("use_consistency", True)),
                    use_contrastive=bool(cfg.get("ssl", {}).get("use_contrastive", True)),
                )
            else:
                assert extra_head is not None
                m = hybrid_step(
                    backbone=backbone,
                    physio_head=head,
                    wave_head=extra_head,
                    x=x,
                    fs=int(cfg["data"].get("fs", 500)),
                    lambda_physio=float(cfg.get("ssl", {}).get("lambda_physio", 1.0)),
                    lambda_wave=float(cfg.get("ssl", {}).get("lambda_wave", 1.0)),
                    perturb_mode=str(cfg.get("ssl", {}).get("perturb_mode", "mixed")),
                )

            loss = m["loss"]
            loss.backward()
            optim.step()

            scalars = _scalar_metrics(m)
            train_logs.append(scalars)
            pbar.set_postfix(loss=f"{scalars['loss']:.4f}")

        train_epoch = pd.DataFrame(train_logs).mean(numeric_only=True).to_dict()

        backbone.eval()
        head.eval()
        if extra_head is not None:
            extra_head.eval()

        val_logs = []
        with torch.no_grad():
            for batch in loaders.val:
                x = batch["x"].to(device)
                if method == "ppt_classic":
                    m = _ppt_step(backbone=backbone, head=head, x=x, cfg=cfg)
                elif method == "wavepuzzle":
                    m = wavepuzzle_step(backbone=backbone, head=head, x=x, fs=int(cfg["data"].get("fs", 500)))
                elif method == "physio_ppt":
                    m = physio_ppt_step(
                        backbone=backbone,
                        head=head,
                        x=x,
                        fs=int(cfg["data"].get("fs", 500)),
                        pre_ms=int(cfg.get("ssl", {}).get("beat_pre_ms", 200)),
                        perturb_mode=str(cfg.get("ssl", {}).get("perturb_mode", "mixed")),
                        use_consistency=bool(cfg.get("ssl", {}).get("use_consistency", True)),
                        use_contrastive=bool(cfg.get("ssl", {}).get("use_contrastive", True)),
                    )
                else:
                    assert extra_head is not None
                    m = hybrid_step(
                        backbone=backbone,
                        physio_head=head,
                        wave_head=extra_head,
                        x=x,
                        fs=int(cfg["data"].get("fs", 500)),
                    )
                val_logs.append(_scalar_metrics(m))

        val_epoch = pd.DataFrame(val_logs).mean(numeric_only=True).to_dict()
        row = {
            "epoch": epoch,
            **{f"train_{k}": float(v) for k, v in train_epoch.items()},
            **{f"val_{k}": float(v) for k, v in val_epoch.items()},
        }
        metrics_rows.append(row)
        jsonl.log("epoch_end", row)

        val_loss = float(val_epoch.get("loss", np.inf))
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "extra_head": extra_head.state_dict() if extra_head is not None else None,
                "config": cfg,
                "seed": seed,
                "epoch": epoch,
                "best_val_loss": best_val,
            }
            torch.save(ckpt, run_dir / "best_pretrain.pt")

        logger.info(
            "pretrain epoch=%d method=%s train_loss=%.4f val_loss=%.4f",
            epoch,
            method,
            float(train_epoch.get("loss", np.nan)),
            float(val_epoch.get("loss", np.nan)),
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    jsonl.close()

    return {
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best_pretrain.pt"),
        "metrics_csv": str(run_dir / "metrics.csv"),
        "method": method,
        "seed": seed,
    }


__all__ = ["run_pretrain", "build_backbone"]
