"""Beat-Order Score implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..data.beat_segment import beat_segment_slices
from .acf_cos import autocorrelation_1d, cosine_similarity


def beat_order_score_single(beat: np.ndarray, fs: int = 500, pre_ms: int = 200, max_lag: int = 80) -> float:
    """Estimate beat order consistency using segment autocorrelation relations.

    A simple robust metric for short papers:
    - Compute ACF for P/QRS/T mean signals.
    - Score is average cosine(P, QRS) + cosine(QRS, T) - cosine(P, T).
    """
    if beat.ndim != 2:
        raise AssertionError(f"Expected beat shape [C, T], got {beat.shape}")

    seg = beat_segment_slices(beat_len=beat.shape[1], fs=fs, pre_ms=pre_ms)
    p = beat[:, seg["P"]].mean(axis=0)
    qrs = beat[:, seg["QRS"]].mean(axis=0)
    t = beat[:, seg["T"]].mean(axis=0)

    acf_p = autocorrelation_1d(p, max_lag=max_lag)
    acf_qrs = autocorrelation_1d(qrs, max_lag=max_lag)
    acf_t = autocorrelation_1d(t, max_lag=max_lag)

    s1 = cosine_similarity(acf_p, acf_qrs)
    s2 = cosine_similarity(acf_qrs, acf_t)
    s3 = cosine_similarity(acf_p, acf_t)
    return float((s1 + s2 - s3) / 2.0)


def analyze_beats_npz(npz_path: str | Path, fs: int = 500, pre_ms: int = 200) -> pd.DataFrame:
    """Compute Beat-Order Score for all beats in NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    beats = data["signals"]

    rows = []
    for i in range(beats.shape[0]):
        score = beat_order_score_single(np.asarray(beats[i]), fs=fs, pre_ms=pre_ms)
        rows.append({"sample_idx": i, "beat_order_score": score})
    return pd.DataFrame(rows)


def aggregate_beat_order_score(df: pd.DataFrame) -> Dict[str, float]:
    """Summarize beat-order score statistics."""
    return {
        "bos_mean": float(df["beat_order_score"].mean()),
        "bos_std": float(df["beat_order_score"].std()),
        "n_samples": int(df.shape[0]),
    }
