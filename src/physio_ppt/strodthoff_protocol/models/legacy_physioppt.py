"""Compatibility wrapper for loading legacy Physio-PPT backbones."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from physio_ppt.experiments.train_pretrain import build_backbone


class LegacyPhysioPPTClassifier(nn.Module):
    """Backbone from legacy pretraining configs + linear multilabel head."""

    def __init__(self, backbone_cfg: Dict[str, object], num_classes: int) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        feat_dim = int(getattr(self.backbone, "out_dim"))
        self.head = nn.Linear(feat_dim, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        pooled = out["pooled"]
        return self.head(pooled)

