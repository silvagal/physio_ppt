"""ACF-COS orderness analysis utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def autocorrelation_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation up to max_lag (inclusive).

    Lags beyond the signal length are set to zero instead of raising a
    broadcast error (which would occur because ``x0[:n-lag]`` becomes a
    negative-index slice while ``x0[lag:]`` is empty).
    """
    if x.ndim != 1:
        raise AssertionError(f"Expected 1D signal, got {x.shape}")
    x0 = x - np.mean(x)
    var = np.var(x0)
    if var < 1e-8:
        return np.zeros((max_lag + 1,), dtype=np.float32)

    n = x0.shape[0]
    acf = np.zeros((max_lag + 1,), dtype=np.float64)
    for lag in range(min(max_lag + 1, n)):  # guard: lag must be < n
        a = x0[: n - lag]
        b = x0[lag:]
        acf[lag] = float(np.mean(a * b) / var)
    # lags >= n remain zero (no valid samples to estimate autocorrelation)
    return acf.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with numerical guard."""
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-8:
        return 0.0
    return float(np.dot(a, b) / den)


def _permute_time_chunks(x: np.ndarray, chunk: int = 40) -> np.ndarray:
    """Numpy chunk permutation over time axis for [C, T]."""
    c, t = x.shape
    n = t // chunk
    if n < 2:
        return x.copy()
    head = x[:, : n * chunk].reshape(c, n, chunk)
    perm = np.random.permutation(n)
    head = head[:, perm, :].reshape(c, n * chunk)
    tail = x[:, n * chunk :]
    return np.concatenate([head, tail], axis=1)


def acf_cos_sample(signal: np.ndarray, max_lag: int = 200, chunk: int = 40) -> np.ndarray:
    """Compute per-lead ACF-COS between original and permuted signal."""
    if signal.ndim != 2:
        raise AssertionError(f"Expected [C, T], got {signal.shape}")
    out = []
    perm = _permute_time_chunks(signal, chunk=chunk)
    for ch in range(signal.shape[0]):
        acf_o = autocorrelation_1d(signal[ch], max_lag=max_lag)
        acf_p = autocorrelation_1d(perm[ch], max_lag=max_lag)
        out.append(cosine_similarity(acf_o, acf_p))
    return np.asarray(out, dtype=np.float32)


def analyze_npz_signals(npz_path: str | Path, max_lag: int = 200, chunk: int = 40) -> pd.DataFrame:
    """Run ACF-COS on every sample in an NPZ signal file."""
    data = np.load(npz_path, allow_pickle=True)
    signals = data["signals"]

    rows = []
    for i in range(signals.shape[0]):
        score = acf_cos_sample(np.asarray(signals[i]), max_lag=max_lag, chunk=chunk)
        rows.append(
            {
                "sample_idx": i,
                "acf_cos_mean": float(np.mean(score)),
                "acf_cos_std": float(np.std(score)),
                "n_leads": int(score.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_acf_cos(df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate per-sample ACF-COS into dataset summary."""
    return {
        "acf_cos_mean": float(df["acf_cos_mean"].mean()),
        "acf_cos_std": float(df["acf_cos_mean"].std()),
        "n_samples": int(df.shape[0]),
    }


def spearman_orderness_gain(orderness: np.ndarray, gains: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman correlation between orderness and gains."""
    from scipy.stats import spearmanr

    corr, p = spearmanr(orderness, gains)
    return float(corr), float(p)
