"""Window-to-record aggregation."""
from __future__ import annotations

import torch


def aggregate_logits(window_logits: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Aggregate window logits into one record-level logit vector.

    Parameters
    ----------
    window_logits:
        Tensor shape (N_windows, num_classes).
    """
    if window_logits.ndim != 2:
        raise ValueError(f"Expected (N, C), got {tuple(window_logits.shape)}")
    mode_l = mode.lower()
    if mode_l == "mean":
        return window_logits.mean(dim=0)
    if mode_l == "max":
        return window_logits.max(dim=0).values
    if mode_l == "median":
        return window_logits.median(dim=0).values
    raise ValueError(f"Unsupported aggregation mode: {mode}")

