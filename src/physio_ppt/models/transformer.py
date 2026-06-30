"""Patch-based Transformer encoder for multi-lead ECG."""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from .pooling import AttentivePooling, MeanPooling


def sinusoidal_positional_encoding(n_tokens: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Build sinusoidal positional encoding [1, N, D]."""
    pos = torch.arange(n_tokens, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros((n_tokens, d_model), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class PatchEmbed1D(nn.Module):
    """Turn [B, C, T] signals into patch tokens."""

    def __init__(self, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.proj = nn.Linear(self.patch_len, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if x.ndim != 3:
            raise AssertionError(f"Expected [B, C, T], got {tuple(x.shape)}")
        b, c, t = x.shape
        if t < self.patch_len:
            pad = self.patch_len - t
            x = torch.nn.functional.pad(x, (0, pad))
            t = x.shape[-1]

        n = 1 + (t - self.patch_len) // self.stride
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, N, P]
        if patches.shape[2] != n:
            raise AssertionError("Unexpected patch count mismatch")

        patches = patches.contiguous().view(b, c * n, self.patch_len)
        tokens = self.proj(patches)
        return tokens, n


class ECGTransformerEncoder(nn.Module):
    """Patch Transformer with optional channel embeddings and pooling."""

    def __init__(
        self,
        in_channels: int,
        patch_len: int = 125,
        stride: int = 62,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
        learnable_positional: bool = False,
        max_tokens: int = 4096,
        use_channel_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed = PatchEmbed1D(patch_len=patch_len, stride=stride, d_model=d_model)
        self.use_channel_embedding = bool(use_channel_embedding)

        if self.use_channel_embedding:
            self.channel_embed = nn.Embedding(self.in_channels, d_model)
        else:
            self.channel_embed = None

        self.learnable_positional = bool(learnable_positional)
        if self.learnable_positional:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if pooling == "mean":
            self.pool = MeanPooling()
        elif pooling == "attn":
            self.pool = AttentivePooling(d_model)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

    @property
    def out_dim(self) -> int:
        return self.encoder.layers[0].self_attn.embed_dim  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise AssertionError(f"Expected [B, C, T], got {tuple(x.shape)}")
        b, c, _ = x.shape
        if c != self.in_channels:
            raise AssertionError(f"Expected {self.in_channels} channels, got {c}")

        tokens, n_per_channel = self.embed(x)  # [B, C*N, D]
        n_tokens = tokens.shape[1]

        if self.use_channel_embedding and self.channel_embed is not None:
            chan_idx = torch.arange(c, device=x.device).view(1, c, 1).repeat(1, 1, n_per_channel)
            chan_idx = chan_idx.view(1, c * n_per_channel)
            tokens = tokens + self.channel_embed(chan_idx)

        if self.learnable_positional and self.pos_embed is not None:
            if n_tokens > self.pos_embed.shape[1]:
                raise ValueError(f"n_tokens={n_tokens} exceeds max positional tokens={self.pos_embed.shape[1]}")
            tokens = tokens + self.pos_embed[:, :n_tokens, :]
        else:
            tokens = tokens + sinusoidal_positional_encoding(n_tokens, tokens.shape[-1], tokens.device)

        enc = self.encoder(tokens)
        pooled = self.pool(enc)
        return {"tokens": enc, "pooled": pooled}
