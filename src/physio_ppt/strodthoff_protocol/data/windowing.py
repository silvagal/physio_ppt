"""Window extraction utilities for record-level ECG evaluation."""
from __future__ import annotations

from typing import List

import torch


def compute_stride(window_size: int, overlap: float) -> int:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    stride = int(round(window_size * (1.0 - overlap)))
    return max(1, stride)


def sliding_window_starts(signal_len: int, window_size: int, overlap: float) -> List[int]:
    if signal_len <= 0:
        raise ValueError("signal_len must be positive")
    if signal_len <= window_size:
        return [0]
    stride = compute_stride(window_size, overlap)
    starts: List[int] = list(range(0, signal_len - window_size + 1, stride))
    last = signal_len - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_windows(signal: torch.Tensor, window_size: int, overlap: float) -> torch.Tensor:
    """Extract sliding windows from one ECG record.

    Parameters
    ----------
    signal:
        Tensor shape (C, T).
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected signal shape (C, T), got {tuple(signal.shape)}")
    channels, total_len = signal.shape
    if total_len < window_size:
        pad = window_size - total_len
        signal = torch.nn.functional.pad(signal, (0, pad), mode="replicate")
        total_len = window_size

    starts = sliding_window_starts(total_len, window_size, overlap)
    windows = [signal[:, s : s + window_size] for s in starts]
    if not windows:
        return signal.new_zeros((0, channels, window_size))
    return torch.stack(windows, dim=0)


def sample_random_windows(batch_signals: torch.Tensor, window_size: int, generator: torch.Generator) -> torch.Tensor:
    """Sample one random window per record for training.

    Parameters
    ----------
    batch_signals:
        Tensor shape (B, C, T).
    """
    if batch_signals.ndim != 3:
        raise ValueError(f"Expected batch shape (B, C, T), got {tuple(batch_signals.shape)}")
    batch, channels, total_len = batch_signals.shape
    if total_len < window_size:
        pad = window_size - total_len
        batch_signals = torch.nn.functional.pad(batch_signals, (0, pad), mode="replicate")
        total_len = window_size

    if total_len == window_size:
        return batch_signals

    max_start = total_len - window_size
    starts = torch.randint(
        low=0,
        high=max_start + 1,
        size=(batch,),
        generator=generator,
        device=batch_signals.device,
    )
    out = batch_signals.new_empty((batch, channels, window_size))
    for i in range(batch):
        s = int(starts[i].item())
        out[i] = batch_signals[i, :, s : s + window_size]
    return out

