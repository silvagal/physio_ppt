"""Bootstrap helpers for test-set uncertainty estimates."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def bootstrap_metric_ci(
    y_true: np.ndarray,
    logits: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    if y_true.shape != logits.shape:
        raise ValueError("y_true and logits must have same shape")
    n = y_true.shape[0]
    rng = np.random.default_rng(int(seed))
    vals = np.zeros((int(n_bootstrap),), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(metric_fn(y_true[idx], logits[idx]))
    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "ci_low": lo,
        "ci_high": hi,
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
    }

