"""ECGWavePuzzle task implementation."""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ..data.beat_segment import beat_segment_slices
from ..models.losses import info_nce_loss
from .ppt_views import weak_view


PERMUTATIONS: List[Tuple[str, str, str]] = list(itertools.permutations(["P", "QRS", "T"]))


def make_wavepuzzle_batch(x: torch.Tensor, fs: int, pre_ms: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Create jigsawed beat batch and permutation labels."""
    if x.ndim != 3:
        raise AssertionError(f"Expected [B, C, T], got {tuple(x.shape)}")

    slices = beat_segment_slices(beat_len=x.shape[-1], fs=fs, pre_ms=pre_ms)
    out = torch.empty_like(x)
    labels = torch.empty((x.shape[0],), dtype=torch.long, device=x.device)

    for i in range(x.shape[0]):
        pidx = int(torch.randint(0, len(PERMUTATIONS), (1,), device=x.device).item())
        order = PERMUTATIONS[pidx]
        chunks = [x[i : i + 1, :, slices[name]] for name in order]
        out[i] = torch.cat(chunks, dim=-1)[0]
        labels[i] = pidx

    return out, labels


def wavepuzzle_step(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    x: torch.Tensor,
    fs: int,
    lambda_contrast: float = 0.2,
    temperature: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Single forward step for wavepuzzle task."""
    x_perm, y_perm = make_wavepuzzle_batch(x, fs=fs)
    feat_perm = backbone(x_perm)["pooled"]
    out_perm = head(feat_perm)
    loss_perm = F.cross_entropy(out_perm["perm_logits"], y_perm)

    loss = loss_perm
    metrics: Dict[str, torch.Tensor] = {
        "loss_perm": loss_perm.detach(),
        "acc_perm": (out_perm["perm_logits"].argmax(dim=1) == y_perm).float().mean().detach(),
    }

    if lambda_contrast > 0.0:
        v1 = weak_view(x)
        v2 = weak_view(x)
        z1 = head(backbone(v1)["pooled"])["projection"]
        z2 = head(backbone(v2)["pooled"])["projection"]
        loss_ctr = info_nce_loss(z1, z2, temperature=temperature)
        loss = loss + lambda_contrast * loss_ctr
        metrics["loss_contrast"] = loss_ctr.detach()

    metrics["loss"] = loss
    return metrics
