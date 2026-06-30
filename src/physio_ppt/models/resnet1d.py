"""ResNet1D backbone for ECG sequences."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """Basic residual block for 1D signals."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.act(out + identity)
        return out


class ResNet1DEncoder(nn.Module):
    """Compact ResNet1D encoder with global pooling."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        d_model: int = 256,
        blocks_per_stage: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)

        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, base_channels, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stage1 = self._make_stage(c1, c1, blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(c1, c2, blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(c2, c3, blocks_per_stage[2], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c3, d_model)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    @property
    def out_dim(self) -> int:
        return self.proj.out_features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise AssertionError(f"Expected [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise AssertionError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.pool(h).squeeze(-1)
        z = self.proj(h)
        return {"tokens": h.unsqueeze(1), "pooled": z}
