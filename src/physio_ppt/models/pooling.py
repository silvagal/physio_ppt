"""Pooling layers for token and temporal embeddings."""
from __future__ import annotations

import torch
from torch import nn


class MeanPooling(nn.Module):
    """Simple mean pooling over sequence dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError(f"Expected [B, N, D], got {tuple(x.shape)}")
        return x.mean(dim=1)


class AttentivePooling(nn.Module):
    """Single-head attention pooling over tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError(f"Expected [B, N, D], got {tuple(x.shape)}")
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(dim=1)
