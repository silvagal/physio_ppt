"""R-peak detection with optional NeuroKit2 and deterministic fallback."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def _bandpass(signal: np.ndarray, fs: int, lo_hz: float = 5.0, hi_hz: float = 25.0) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(lo_hz / nyq, 1e-4)
    hi = min(hi_hz / nyq, 0.99)
    b, a = butter(2, [lo, hi], btype="band")
    return filtfilt(b, a, signal)


def _fallback_detector(signal: np.ndarray, fs: int) -> np.ndarray:
    """Simple Pan-Tompkins-like fallback detector."""
    x = _bandpass(signal, fs=fs)
    dx = np.diff(x, prepend=x[0])
    sq = dx**2
    win = max(3, int(0.120 * fs))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    integ = np.convolve(sq, kernel, mode="same")

    min_dist = max(1, int(0.25 * fs))
    height = float(np.percentile(integ, 90) * 0.30)
    peaks, _ = find_peaks(integ, distance=min_dist, height=height)
    return peaks.astype(np.int64)


def detect_rpeaks(
    signal: np.ndarray,
    fs: int,
    prefer_neurokit: bool = True,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Detect R-peaks from a 1D ECG signal.

    Returns
    -------
    peaks:
        Sample indices of detected peaks.
    info:
        Method metadata and fallback status.
    """
    if signal.ndim != 1:
        raise AssertionError(f"Expected 1D signal, got shape {signal.shape}")
    if fs <= 0:
        raise ValueError("fs must be positive")

    if prefer_neurokit:
        try:
            import neurokit2 as nk  # type: ignore

            clean = nk.ecg_clean(signal, sampling_rate=fs)
            _, peaks = nk.ecg_peaks(clean, sampling_rate=fs)
            r = np.asarray(peaks.get("ECG_R_Peaks", []), dtype=np.int64)
            if r.size > 0:
                return r, {"method": "neurokit2", "fallback": "false"}
        except Exception:
            pass

    peaks = _fallback_detector(signal, fs=fs)
    return peaks, {"method": "pan_tompkins_fallback", "fallback": "true"}
