"""Augmentations and permutations for PPT-like ECG SSL."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _rand_like(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return (high - low) * torch.rand((x.shape[0], 1, 1), device=x.device) + low


def add_jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Add Gaussian jitter."""
    return x + sigma * torch.randn_like(x)


def scale_amplitude(x: torch.Tensor, low: float = 0.9, high: float = 1.1) -> torch.Tensor:
    """Random global amplitude scaling per sample."""
    return x * _rand_like(x, low=low, high=high)


def random_crop_aligned(x: torch.Tensor, crop_ratio: float = 0.95) -> torch.Tensor:
    """Random crop and resize back to original length."""
    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError("crop_ratio must be in (0, 1]")
    if crop_ratio >= 1.0:
        return x

    b, c, t = x.shape
    crop_len = max(4, int(round(t * crop_ratio)))
    out = torch.empty_like(x)
    for i in range(b):
        start = int(torch.randint(0, t - crop_len + 1, (1,), device=x.device).item())
        seg = x[i : i + 1, :, start : start + crop_len]
        seg = torch.nn.functional.interpolate(seg, size=t, mode="linear", align_corners=False)
        out[i] = seg[0]
    return out


def random_time_mask(x: torch.Tensor, max_ratio: float = 0.1) -> torch.Tensor:
    """Mask a random temporal region."""
    b, c, t = x.shape
    out = x.clone()
    max_len = max(1, int(round(max_ratio * t)))
    for i in range(b):
        m = int(torch.randint(1, max_len + 1, (1,), device=x.device).item())
        s = int(torch.randint(0, t - m + 1, (1,), device=x.device).item())
        out[i, :, s : s + m] = 0.0
    return out


def random_time_warp(x: torch.Tensor, warp_scale: float = 0.08) -> torch.Tensor:
    """Mild differentiable time warp via piecewise interpolation."""
    b, c, t = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        alpha = float(np.clip(np.random.normal(loc=1.0, scale=warp_scale), 0.85, 1.15))
        new_t = max(8, int(round(t * alpha)))
        warped = torch.nn.functional.interpolate(x[i : i + 1], size=new_t, mode="linear", align_corners=False)
        warped = torch.nn.functional.interpolate(warped, size=t, mode="linear", align_corners=False)
        out[i] = warped[0]
    return out


def _permute_chunks_1d(sig: torch.Tensor, patch_len: int) -> torch.Tensor:
    """Permute non-overlapping chunks of a 1D signal [C, T]."""
    c, t = sig.shape
    n = t // patch_len
    if n < 2:
        return sig
    head = sig[:, : n * patch_len].view(c, n, patch_len)
    tail = sig[:, n * patch_len :]
    perm = torch.randperm(n, device=sig.device)
    head = head[:, perm, :].reshape(c, n * patch_len)
    if tail.numel() == 0:
        return head
    return torch.cat([head, tail], dim=-1)


def permute_global_patches(x: torch.Tensor, patch_len: int = 40) -> torch.Tensor:
    """Permute temporal chunks across the whole signal."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = _permute_chunks_1d(x[i], patch_len=patch_len)
    return out


def permute_channels(x: torch.Tensor) -> torch.Tensor:
    """Randomly permute channel order per sample."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        perm = torch.randperm(x.shape[1], device=x.device)
        out[i] = x[i, perm]
    return out


def permute_within_segments(
    x: torch.Tensor,
    segment_slices: Dict[str, slice],
    patch_len: int = 12,
) -> torch.Tensor:
    """Permute patches only within P, QRS, T segments."""
    out = x.clone()
    for seg, sl in segment_slices.items():
        if sl.stop is None or sl.start is None:
            continue
        if sl.stop - sl.start < 2:
            continue
        seg_x = out[:, :, sl]
        seg_perm = permute_global_patches(seg_x, patch_len=patch_len)
        out[:, :, sl] = seg_perm
    return out


def permute_segments(x: torch.Tensor, segment_slices: Dict[str, slice], order: tuple[str, str, str]) -> torch.Tensor:
    """Reorder P/QRS/T macro-segments according to a target order."""
    chunks = []
    for name in order:
        sl = segment_slices[name]
        chunks.append(x[:, :, sl])
    return torch.cat(chunks, dim=-1)


def weak_view(x: torch.Tensor) -> torch.Tensor:
    """Weak view: mild augmentations preserving physiological order."""
    out = x
    out = add_jitter(out, sigma=0.01)
    out = scale_amplitude(out, low=0.95, high=1.05)
    out = random_crop_aligned(out, crop_ratio=0.97)
    out = random_time_mask(out, max_ratio=0.05)
    return out


def strong_view_ppt(x: torch.Tensor, patch_len: int = 40, channel_perm_prob: float = 0.3) -> torch.Tensor:
    """Strong PPT view with global patch permutation and optional channel shuffle."""
    out = weak_view(x)
    out = add_jitter(out, sigma=0.02)
    out = random_time_warp(out, warp_scale=0.10)
    out = permute_global_patches(out, patch_len=patch_len)
    if torch.rand(1).item() < channel_perm_prob:
        out = permute_channels(out)
    return out
