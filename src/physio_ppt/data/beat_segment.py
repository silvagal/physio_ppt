"""Beat extraction and P/QRS/T segment utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SegmentBounds:
    """Fixed P/QRS/T boundaries in beat sample coordinates."""

    p_start: int
    p_end: int
    qrs_start: int
    qrs_end: int
    t_start: int
    t_end: int


def fixed_segment_bounds(beat_len: int, fs: int, r_index: int) -> SegmentBounds:
    """Construct robust fallback segment boundaries relative to R peak."""
    if beat_len <= 0:
        raise ValueError("beat_len must be > 0")

    p_start = max(0, r_index - int(0.20 * fs))
    p_end = max(p_start + 1, r_index - int(0.04 * fs))

    qrs_start = max(0, r_index - int(0.04 * fs))
    qrs_end = min(beat_len, r_index + int(0.08 * fs))

    t_start = min(beat_len - 1, qrs_end + int(0.02 * fs))
    t_end = min(beat_len, r_index + int(0.32 * fs))

    if t_end <= t_start:
        t_end = min(beat_len, t_start + max(1, int(0.08 * fs)))

    return SegmentBounds(
        p_start=int(p_start),
        p_end=int(p_end),
        qrs_start=int(qrs_start),
        qrs_end=int(qrs_end),
        t_start=int(t_start),
        t_end=int(t_end),
    )


def extract_beats(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    fs: int,
    pre_ms: int = 200,
    post_ms: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract fixed windows around R-peaks.

    Parameters
    ----------
    signal:
        ECG array with shape [C, T].
    rpeaks:
        R-peak positions in samples.

    Returns
    -------
    beats:
        Array [N, C, L].
    kept_rpeaks:
        R-peak indices retained after boundary checks.
    """
    if signal.ndim != 2:
        raise AssertionError(f"Expected [C, T], got {signal.shape}")

    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))
    length = pre + post
    if length <= 0:
        raise ValueError("Invalid beat window length")

    beats: List[np.ndarray] = []
    kept: List[int] = []
    t_total = signal.shape[1]

    for r in rpeaks.astype(np.int64):
        start = int(r - pre)
        end = int(r + post)
        if start < 0 or end > t_total:
            continue
        beat = signal[:, start:end]
        if beat.shape[1] != length:
            continue
        beats.append(beat.astype(np.float32))
        kept.append(int(r))

    if not beats:
        return np.zeros((0, signal.shape[0], length), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(beats, axis=0), np.asarray(kept, dtype=np.int64)


def beat_segment_slices(
    beat_len: int,
    fs: int,
    pre_ms: int = 200,
) -> Dict[str, slice]:
    """Return contiguous, non-overlapping slices that partition [0, beat_len).

    The returned slices P / QRS / T together cover the entire beat so that
    ``permute_segments`` can safely reorder them without changing the signal
    length.  Specifically:

    - P   : [0, qrs_start)          – pre-QRS region (baseline + P wave)
    - QRS : [qrs_start, qrs_end)    – ventricular depolarisation
    - T   : [qrs_end, beat_len)     – repolarisation + post-beat baseline
    """
    r_idx = int(round(pre_ms * fs / 1000.0))
    bounds = fixed_segment_bounds(beat_len=beat_len, fs=fs, r_index=r_idx)
    # Use QRS boundaries as the pivot; P absorbs leading baseline and T
    # absorbs the post-T tail so that P + QRS + T == beat_len exactly.
    return {
        "P": slice(0, bounds.qrs_start),
        "QRS": slice(bounds.qrs_start, bounds.qrs_end),
        "T": slice(bounds.qrs_end, beat_len),
    }


def delineate_bounds_optional(beat_1d: np.ndarray, fs: int, r_index: int) -> Optional[SegmentBounds]:
    """Try NeuroKit2 delineation and return bounds if valid."""
    try:
        import neurokit2 as nk  # type: ignore

        clean = nk.ecg_clean(beat_1d, sampling_rate=fs)
        _, waves = nk.ecg_delineate(
            clean,
            rpeaks=np.asarray([r_index]),
            sampling_rate=fs,
            method="dwt",
            show=False,
            show_type="all",
        )

        p_on = waves.get("ECG_P_Onsets", [np.nan])[0]
        p_off = waves.get("ECG_P_Offsets", [np.nan])[0]
        qrs_on = waves.get("ECG_R_Onsets", [np.nan])[0]
        qrs_off = waves.get("ECG_R_Offsets", [np.nan])[0]
        t_on = waves.get("ECG_T_Onsets", [np.nan])[0]
        t_off = waves.get("ECG_T_Offsets", [np.nan])[0]

        arr = np.asarray([p_on, p_off, qrs_on, qrs_off, t_on, t_off], dtype=np.float64)
        if np.any(~np.isfinite(arr)):
            return None

        vals = arr.astype(np.int64)
        if not (0 <= vals[0] < vals[1] <= beat_1d.shape[0]):
            return None
        if not (0 <= vals[2] < vals[3] <= beat_1d.shape[0]):
            return None
        if not (0 <= vals[4] < vals[5] <= beat_1d.shape[0]):
            return None

        return SegmentBounds(
            p_start=int(vals[0]),
            p_end=int(vals[1]),
            qrs_start=int(vals[2]),
            qrs_end=int(vals[3]),
            t_start=int(vals[4]),
            t_end=int(vals[5]),
        )
    except Exception:
        return None


def segment_bounds_with_fallback(
    beat: np.ndarray,
    fs: int,
    r_index: int,
    prefer_delineation: bool = True,
) -> Tuple[SegmentBounds, bool]:
    """Get segment bounds using delineation when possible, otherwise fixed offsets.

    Returns
    -------
    bounds:
        Segment boundaries.
    used_fallback:
        True when fixed-offset fallback was used.
    """
    if beat.ndim != 2:
        raise AssertionError(f"Expected beat [C, T], got {beat.shape}")

    if prefer_delineation:
        b = delineate_bounds_optional(beat[0], fs=fs, r_index=r_index)
        if b is not None:
            return b, False

    return fixed_segment_bounds(beat_len=beat.shape[1], fs=fs, r_index=r_index), True
