"""Physio-PPT task with physiology-constrained perturbations."""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ..data.beat_segment import beat_segment_slices
from ..models.losses import info_nce_loss
from .ppt_views import (
    add_jitter,
    permute_segments,
    permute_within_segments,
    random_crop_aligned,
    random_time_mask,
    random_time_warp,
    scale_amplitude,
)


SEG_ORDERS: List[Tuple[str, str, str]] = list(itertools.permutations(["P", "QRS", "T"]))


def weak_physio_view(x: torch.Tensor) -> torch.Tensor:
    """Weak view aligned with physiological morphology."""
    out = x
    out = add_jitter(out, sigma=0.01)
    out = scale_amplitude(out, low=0.95, high=1.05)
    out = random_crop_aligned(out, crop_ratio=0.98)
    out = random_time_mask(out, max_ratio=0.04)
    return out


def strong_physio_view(
    x: torch.Tensor,
    segment_slices: Dict[str, slice],
    intra_patch_len: int = 12,
) -> torch.Tensor:
    """Strong view with constrained intra-segment patch permutation."""
    out = weak_physio_view(x)
    out = random_time_warp(out, warp_scale=0.08)
    out = permute_within_segments(out, segment_slices=segment_slices, patch_len=intra_patch_len)
    return out


def make_physio_perturbation(
    x: torch.Tensor,
    fs: int,
    pre_ms: int,
    perturb_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate perturbation and optional segment-order labels.

    perturb_mode
    ------------
    - "intra": permute within segments only.
    - "inter": swap macro segments P/QRS/T.
    - "mixed": random choice between intra/inter per sample.
    """
    slices = beat_segment_slices(beat_len=x.shape[-1], fs=fs, pre_ms=pre_ms)
    b = x.shape[0]

    out = torch.empty_like(x)
    order_labels = torch.full((b,), -1, device=x.device, dtype=torch.long)

    for i in range(b):
        mode_i = perturb_mode
        if perturb_mode == "mixed":
            mode_i = "inter" if torch.rand(1).item() < 0.5 else "intra"

        if mode_i == "intra":
            out[i : i + 1] = permute_within_segments(x[i : i + 1], segment_slices=slices)
            order_labels[i] = 0
        elif mode_i == "inter":
            pidx = int(torch.randint(0, len(SEG_ORDERS), (1,), device=x.device).item())
            order = SEG_ORDERS[pidx]
            out[i : i + 1] = permute_segments(x[i : i + 1], segment_slices=slices, order=order)
            order_labels[i] = pidx
        else:
            raise ValueError(f"Unsupported perturb_mode: {perturb_mode}")

    return out, order_labels


def physio_ppt_step(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    x: torch.Tensor,
    fs: int,
    pre_ms: int = 200,
    lambda_consistency: float = 1.0,
    lambda_contrast: float = 1.0,
    lambda_segment: float = 0.2,
    perturb_mode: str = "mixed",
    temperature: float = 0.2,
    use_consistency: bool = True,
    use_contrastive: bool = True,
) -> Dict[str, torch.Tensor]:
    """Single Physio-PPT optimization step."""
    seg_slices = beat_segment_slices(beat_len=x.shape[-1], fs=fs, pre_ms=pre_ms)
    weak = weak_physio_view(x)
    strong = strong_physio_view(x, segment_slices=seg_slices)

    feat_w = backbone(weak)["pooled"]
    feat_s = backbone(strong)["pooled"]
    out_w = head(feat_w)
    out_s = head(feat_s)

    total = torch.tensor(0.0, device=x.device)
    metrics: Dict[str, torch.Tensor] = {}

    if use_contrastive:
        loss_ctr = info_nce_loss(out_w["projection"], out_s["projection"], temperature=temperature)
        total = total + lambda_contrast * loss_ctr
        metrics["loss_contrast"] = loss_ctr.detach()

    if use_consistency:
        perturbed, order_labels = make_physio_perturbation(x, fs=fs, pre_ms=pre_ms, perturb_mode=perturb_mode)
        feat_p = backbone(perturbed)["pooled"]
        out_p = head(feat_p)

        logits_pos = out_w["consistency_logits"]
        logits_neg = out_p["consistency_logits"]
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        targets = torch.cat(
            [
                torch.ones_like(logits_pos),
                torch.zeros_like(logits_neg),
            ],
            dim=0,
        )
        loss_cons = F.binary_cross_entropy_with_logits(logits, targets)
        total = total + lambda_consistency * loss_cons
        pred_bin = (torch.sigmoid(logits) >= 0.5).float()
        metrics["loss_consistency"] = loss_cons.detach()
        metrics["acc_consistency"] = (pred_bin == targets).float().mean().detach()

        if "segment_logits" in out_p and lambda_segment > 0.0:
            valid = order_labels >= 0
            if valid.any():
                loss_seg = F.cross_entropy(out_p["segment_logits"][valid], order_labels[valid])
                total = total + lambda_segment * loss_seg
                acc_seg = (
                    out_p["segment_logits"][valid].argmax(dim=1) == order_labels[valid]
                ).float().mean()
                metrics["loss_segment"] = loss_seg.detach()
                metrics["acc_segment"] = acc_seg.detach()

    metrics["loss"] = total
    return metrics
