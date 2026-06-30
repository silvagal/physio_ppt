"""Hybrid multi-task objective: Physio-PPT + WavePuzzle."""
from __future__ import annotations

from typing import Dict

import torch

from .physio_ppt_task import physio_ppt_step
from .wavepuzzle_task import wavepuzzle_step


def hybrid_step(
    backbone: torch.nn.Module,
    physio_head: torch.nn.Module,
    wave_head: torch.nn.Module,
    x: torch.Tensor,
    fs: int,
    lambda_physio: float = 1.0,
    lambda_wave: float = 1.0,
    perturb_mode: str = "mixed",
) -> Dict[str, torch.Tensor]:
    """Compute weighted multi-task loss and metrics."""
    m_phys = physio_ppt_step(
        backbone=backbone,
        head=physio_head,
        x=x,
        fs=fs,
        perturb_mode=perturb_mode,
    )
    m_wave = wavepuzzle_step(
        backbone=backbone,
        head=wave_head,
        x=x,
        fs=fs,
    )

    total = lambda_physio * m_phys["loss"] + lambda_wave * m_wave["loss"]
    out: Dict[str, torch.Tensor] = {
        "loss": total,
        "loss_physio": m_phys["loss"].detach(),
        "loss_wave": m_wave["loss"].detach(),
    }

    for k, v in m_phys.items():
        if k != "loss":
            out[f"physio_{k}"] = v.detach() if torch.is_tensor(v) else v
    for k, v in m_wave.items():
        if k != "loss":
            out[f"wave_{k}"] = v.detach() if torch.is_tensor(v) else v

    return out
