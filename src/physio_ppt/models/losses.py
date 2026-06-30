"""Loss functions for supervised and SSL training."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE over positive pairs in a batch."""
    if z1.shape != z2.shape:
        raise AssertionError(f"Shape mismatch: {tuple(z1.shape)} vs {tuple(z2.shape)}")
    if z1.ndim != 2:
        raise AssertionError("Expected [B, D] embeddings")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    b = z1.shape[0]

    logits_12 = torch.matmul(z1, z2.t()) / temperature
    logits_21 = torch.matmul(z2, z1.t()) / temperature
    labels = torch.arange(b, device=z1.device)

    loss = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
    return loss


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE for multilabel classification."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * (1.0 - pt).pow(gamma) * bce
    return focal.mean()


def supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
    pos_weight: Optional[torch.Tensor] = None,
    use_focal: bool = False,
) -> torch.Tensor:
    """Compute supervised loss for multilabel or multiclass tasks."""
    if task_type == "multilabel":
        if use_focal:
            return focal_bce_with_logits(logits, targets)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    if task_type == "multiclass":
        if targets.ndim != 1:
            targets = targets.view(-1)
        return F.cross_entropy(logits, targets.long())
    raise ValueError(f"Unsupported task_type: {task_type}")
