"""Paired bootstrap utilities for significance tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class BootstrapResult:
    """Container for paired bootstrap output."""

    delta_mean: float
    ci_low: float
    ci_high: float
    p_value_two_sided: float


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: MetricFn,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> BootstrapResult:
    """Compute paired bootstrap CI for metric difference A - B.

    Resampling is done over sample indices with replacement.
    """
    if y_true.shape[0] != pred_a.shape[0] or y_true.shape[0] != pred_b.shape[0]:
        raise AssertionError("y_true, pred_a, and pred_b must have same first dimension")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    deltas = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m_a = metric_fn(y_true[idx], pred_a[idx])
        m_b = metric_fn(y_true[idx], pred_b[idx])
        deltas[i] = m_a - m_b

    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    p_two_sided = 2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0))

    return BootstrapResult(
        delta_mean=float(np.mean(deltas)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value_two_sided=float(p_two_sided),
    )


def as_dict(result: BootstrapResult) -> Dict[str, float]:
    """Convert BootstrapResult to serializable dictionary."""
    return {
        "delta_mean": result.delta_mean,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "p_value_two_sided": result.p_value_two_sided,
    }
