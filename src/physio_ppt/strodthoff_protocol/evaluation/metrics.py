"""Metrics for multilabel PTB-XL superdiagnostic evaluation."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    f1_score = None
    roc_auc_score = None


def _require_sklearn() -> None:
    if average_precision_score is None or f1_score is None or roc_auc_score is None:
        raise ImportError("scikit-learn is required for evaluation metrics. Install with `pip install scikit-learn`.")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _safe_auroc_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    _require_sklearn()
    out = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
    for c in range(y_true.shape[1]):
        uniq = np.unique(y_true[:, c])
        if uniq.size < 2:
            continue
        out[c] = float(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return out


def _safe_auprc_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    _require_sklearn()
    out = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
    for c in range(y_true.shape[1]):
        uniq = np.unique(y_true[:, c])
        if uniq.size < 2:
            continue
        out[c] = float(average_precision_score(y_true[:, c], y_prob[:, c]))
    return out


def compute_multilabel_metrics(logits: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    _require_sklearn()
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, C), got {logits.shape}")
    if y_true.shape != logits.shape:
        raise ValueError(f"y_true/logits mismatch: {y_true.shape} vs {logits.shape}")

    y_bin = (y_true > 0.5).astype(np.int32)
    y_prob = sigmoid(logits)
    y_pred = (y_prob >= float(threshold)).astype(np.int32)

    auroc_c = _safe_auroc_per_class(y_bin, y_prob)
    auprc_c = _safe_auprc_per_class(y_bin, y_prob)

    return {
        "macro_auroc": float(np.nanmean(auroc_c)),
        "macro_auprc": float(np.nanmean(auprc_c)),
        "macro_f1": float(f1_score(y_bin, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_bin, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_bin, y_pred, average="samples", zero_division=0)),
    }


def compute_per_class_report(
    logits: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    threshold: float = 0.5,
) -> List[Dict[str, float | str]]:
    _require_sklearn()
    if logits.shape[1] != len(class_names):
        raise ValueError("class_names size must match number of classes")
    y_bin = (y_true > 0.5).astype(np.int32)
    y_prob = sigmoid(logits)
    y_pred = (y_prob >= float(threshold)).astype(np.int32)
    auroc_c = _safe_auroc_per_class(y_bin, y_prob)
    auprc_c = _safe_auprc_per_class(y_bin, y_prob)

    rows: List[Dict[str, float | str]] = []
    for c, name in enumerate(class_names):
        rows.append(
            {
                "class_name": str(name),
                "prevalence": float(np.mean(y_bin[:, c])),
                "auroc": float(auroc_c[c]) if not np.isnan(auroc_c[c]) else float("nan"),
                "auprc": float(auprc_c[c]) if not np.isnan(auprc_c[c]) else float("nan"),
                "f1": float(f1_score(y_bin[:, c], y_pred[:, c], average="binary", zero_division=0)),
            }
        )
    return rows
