"""Smoke tests for model forward passes and SSL steps."""
from __future__ import annotations

import torch

from physio_ppt.models.heads import PhysioPPTHead, PPTHead, WavePuzzleHead
from physio_ppt.models.resnet1d import ResNet1DEncoder
from physio_ppt.models.transformer import ECGTransformerEncoder
from physio_ppt.ssl.physio_ppt_task import physio_ppt_step
from physio_ppt.ssl.wavepuzzle_task import wavepuzzle_step


def test_forward_resnet_heads() -> None:
    x = torch.randn(4, 2, 300)
    backbone = ResNet1DEncoder(in_channels=2, base_channels=16, d_model=64)
    feat = backbone(x)["pooled"]
    assert tuple(feat.shape) == (4, 64)

    ppt = PPTHead(in_dim=64, proj_dim=32)
    out = ppt(feat)
    assert out["projection"].shape == (4, 32)
    assert out["consistency_logits"].shape == (4,)


def test_forward_transformer_and_ssl_steps() -> None:
    x = torch.randn(4, 2, 300)
    backbone = ECGTransformerEncoder(
        in_channels=2,
        patch_len=30,
        stride=15,
        d_model=96,
        nhead=3,
        num_layers=2,
        dim_feedforward=192,
    )
    feat = backbone(x)["pooled"]
    assert feat.shape[0] == 4

    wave_head = WavePuzzleHead(in_dim=96, num_permutations=6, proj_dim=32)
    wave_m = wavepuzzle_step(backbone, wave_head, x, fs=500)
    assert "loss" in wave_m

    phys_head = PhysioPPTHead(in_dim=96, proj_dim=32, enable_segment_order=True)
    phys_m = physio_ppt_step(backbone, phys_head, x, fs=500)
    assert "loss" in phys_m
