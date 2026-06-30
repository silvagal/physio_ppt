"""Transformer baseline with capacity comparable to XResNet1D-101."""
from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_positional(n_tokens: int, d_model: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(n_tokens, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros((1, n_tokens, d_model), device=device, dtype=torch.float32)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


class ECGTransformerStrodthoff(nn.Module):
    """Patch-token Transformer for ECG windows."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        d_model: int = 512,
        num_layers: int = 12,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 10,
        patch_stride: int = 10,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pooling: str = "cls",
        learnable_positional: bool = True,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.pooling = str(pooling)
        self.patch_embed = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(d_model),
            kernel_size=int(patch_size),
            stride=int(patch_stride),
            padding=0,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.learnable_positional = bool(learnable_positional)
        if self.learnable_positional:
            self.pos_embed = nn.Parameter(torch.zeros(1, int(max_tokens), int(d_model)))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        ff_dim = int(round(float(mlp_ratio) * float(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(int(d_model), int(num_classes))
        self.dropout = nn.Dropout(float(attention_dropout))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        if self.learnable_positional and self.pos_embed is not None:
            if n > self.pos_embed.shape[1]:
                raise ValueError(f"n_tokens={n} exceeds max_tokens={self.pos_embed.shape[1]}")
            return x + self.pos_embed[:, :n, :]
        return x + _sinusoidal_positional(n, x.shape[-1], x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} leads, got {x.shape[1]}")
        patches = self.patch_embed(x)  # (B, D, N)
        tokens = patches.transpose(1, 2).contiguous()  # (B, N, D)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self._add_positional(tokens)
        tokens = self.dropout(tokens)
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        if self.pooling == "cls":
            pooled = encoded[:, 0]
        elif self.pooling == "mean":
            pooled = encoded[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        return self.head(pooled)

