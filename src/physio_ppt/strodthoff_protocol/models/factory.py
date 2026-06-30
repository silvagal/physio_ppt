"""Model factory and checkpoint loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from .ecg_transformer_strodthoff import ECGTransformerStrodthoff
from .legacy_physioppt import LegacyPhysioPPTClassifier
from .xresnet1d101 import xresnet1d101


def build_model(model_cfg: Dict[str, object], *, in_channels: int, num_classes: int) -> nn.Module:
    name = str(model_cfg.get("name", "xresnet1d101")).lower()
    if name == "xresnet1d101":
        return xresnet1d101(
            in_channels=in_channels,
            num_classes=num_classes,
            width_factor=float(model_cfg.get("width_factor", 1.0)),
            head_dropout=float(model_cfg.get("head_dropout", 0.0)),
        )
    if name == "ecg_transformer_strodthoff":
        return ECGTransformerStrodthoff(
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=int(model_cfg.get("d_model", 512)),
            num_layers=int(model_cfg.get("num_layers", 12)),
            nhead=int(model_cfg.get("nhead", 8)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            patch_size=int(model_cfg.get("patch_size", 10)),
            patch_stride=int(model_cfg.get("patch_stride", 10)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            attention_dropout=float(model_cfg.get("attention_dropout", 0.1)),
            pooling=str(model_cfg.get("pooling", "cls")),
            learnable_positional=bool(model_cfg.get("learnable_positional", True)),
            max_tokens=int(model_cfg.get("max_tokens", 1024)),
        )
    if name == "legacy_physioppt_backbone":
        backbone_cfg = model_cfg.get("backbone")
        if not isinstance(backbone_cfg, dict):
            raise ValueError("legacy_physioppt_backbone requires model.backbone config")
        return LegacyPhysioPPTClassifier(backbone_cfg=backbone_cfg, num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _filter_compatible_state_dict(module: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    target = module.state_dict()
    kept: Dict[str, torch.Tensor] = {}
    skipped = 0
    for k, v in state.items():
        if k in target and tuple(target[k].shape) == tuple(v.shape):
            kept[k] = v
        else:
            skipped += 1
    return kept, skipped


def load_pretrained_backbone(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    source: str = "physio_ppt_pretrain",
) -> Dict[str, object]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint must be a dict")

    if source == "physio_ppt_pretrain":
        if "backbone" not in ckpt:
            raise KeyError("Expected `backbone` key in Physio-PPT pretrain checkpoint")
        target = model.backbone if hasattr(model, "backbone") else model
        state = ckpt["backbone"]
    elif source == "full_model":
        target = model
        state = ckpt.get("model", ckpt)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if not isinstance(state, dict):
        raise TypeError("checkpoint state must be a dict")
    compatible, skipped = _filter_compatible_state_dict(target, state)
    missing, unexpected = target.load_state_dict(compatible, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_keys": int(len(compatible)),
        "skipped_keys_shape_or_missing": int(skipped),
        "missing_keys_after_load": int(len(missing)),
        "unexpected_keys_after_load": int(len(unexpected)),
    }

