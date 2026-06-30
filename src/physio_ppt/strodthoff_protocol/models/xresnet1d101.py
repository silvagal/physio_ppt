"""1D XResNet-101 classifier for 12-lead ECG windows."""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


def _conv_bn_act(in_ch: int, out_ch: int, kernel_size: int, stride: int) -> nn.Sequential:
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        width = int(planes)
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class XResNet1D(nn.Module):
    """XResNet-style 1D network with ResNet-101 depth."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        layers: Sequence[int],
        *,
        stem_channels: Sequence[int] = (32, 32, 64),
        width_factor: float = 1.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must contain 4 stage depths")
        if len(stem_channels) != 3:
            raise ValueError("stem_channels must contain 3 values")

        c0, c1, c2 = [int(round(x * width_factor)) for x in stem_channels]
        self.stem = nn.Sequential(
            _conv_bn_act(in_channels, c0, kernel_size=5, stride=2),
            _conv_bn_act(c0, c1, kernel_size=3, stride=1),
            _conv_bn_act(c1, c2, kernel_size=3, stride=1),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        planes = [64, 128, 256, 512]
        planes = [int(round(p * width_factor)) for p in planes]
        self.inplanes = c2
        self.layer1 = self._make_layer(planes[0], int(layers[0]), stride=1)
        self.layer2 = self._make_layer(planes[1], int(layers[1]), stride=2)
        self.layer3 = self._make_layer(planes[2], int(layers[2]), stride=2)
        self.layer4 = self._make_layer(planes[3], int(layers[3]), stride=2)

        out_ch = planes[3] * Bottleneck1D.expansion
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=float(head_dropout)) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(out_ch, int(num_classes))

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        out_channels = planes * Bottleneck1D.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers: List[nn.Module] = [Bottleneck1D(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")
        out = self.stem(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).squeeze(-1)
        out = self.dropout(out)
        return self.head(out)


def xresnet1d101(
    in_channels: int,
    num_classes: int,
    *,
    width_factor: float = 1.0,
    head_dropout: float = 0.0,
) -> XResNet1D:
    return XResNet1D(
        in_channels=in_channels,
        num_classes=num_classes,
        layers=(3, 4, 23, 3),
        width_factor=width_factor,
        head_dropout=head_dropout,
    )

