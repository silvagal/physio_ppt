"""Task heads for supervised and SSL objectives."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Two-layer projection MLP for contrastive learning."""

    def __init__(self, in_dim: int, proj_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupervisedHead(nn.Module):
    """Classification head for fine-tuning."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(feat)


class PPTHead(nn.Module):
    """Head for PPT-style consistency and contrastive objectives."""

    def __init__(self, in_dim: int, proj_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.proj = ProjectionHead(in_dim, proj_dim=proj_dim, hidden_dim=hidden_dim)
        self.consistency = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "projection": self.proj(feat),
            "consistency_logits": self.consistency(feat).squeeze(-1),
        }


class WavePuzzleHead(nn.Module):
    """Permutation classification head for ECGWavePuzzle."""

    def __init__(self, in_dim: int, num_permutations: int = 6, proj_dim: int = 128) -> None:
        super().__init__()
        self.perm_classifier = nn.Linear(in_dim, num_permutations)
        self.proj = ProjectionHead(in_dim, proj_dim=proj_dim)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "perm_logits": self.perm_classifier(feat),
            "projection": self.proj(feat),
        }


class PhysioPPTHead(nn.Module):
    """Head for Physio-PPT consistency, contrastive, and segment-order tasks."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        num_segment_orders: int = 6,
        enable_segment_order: bool = True,
    ) -> None:
        super().__init__()
        self.proj = ProjectionHead(in_dim, proj_dim=proj_dim, hidden_dim=hidden_dim)
        self.consistency = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.enable_segment_order = bool(enable_segment_order)
        if self.enable_segment_order:
            self.segment_order = nn.Linear(in_dim, num_segment_orders)
        else:
            self.segment_order = None

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {
            "projection": self.proj(feat),
            "consistency_logits": self.consistency(feat).squeeze(-1),
        }
        if self.segment_order is not None:
            out["segment_logits"] = self.segment_order(feat)
        return out
