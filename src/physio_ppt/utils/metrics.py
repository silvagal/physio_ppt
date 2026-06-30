"""Metrics for multilabel and multiclass ECG tasks."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def _safe_macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    vals = []
    for c in range(y_true.shape[1]):
        uniq = np.unique(y_true[:, c])
        if uniq.size < 2:
            continue
        vals.append(roc_auc_score(y_true[:, c], y_score[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def _safe_macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    vals = []
    for c in range(y_true.shape[1]):
        uniq = np.unique(y_true[:, c])
        if uniq.size < 2:
            continue
        vals.append(average_precision_score(y_true[:, c], y_score[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def classification_metrics(
    logits: np.ndarray,
    y_true: np.ndarray,
    task_type: str,
) -> Dict[str, float]:
    """Compute macro/micro metrics for multilabel or multiclass tasks."""
    if logits.ndim != 2:
        raise AssertionError(f"Expected logits shape [N, C], got {logits.shape}")

    if task_type == "multilabel":
        probs = sigmoid(logits)
        y_bin = (y_true > 0.5).astype(np.int32)
        y_pred = (probs >= 0.5).astype(np.int32)
        metrics = {
            "macro_f1": float(f1_score(y_bin, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_bin, y_pred, average="micro", zero_division=0)),
            "macro_auroc": _safe_macro_auroc(y_bin, probs),
            "macro_auprc": _safe_macro_auprc(y_bin, probs),
        }
        return metrics

    if task_type == "multiclass":
        probs = softmax(logits, axis=1)
        y_idx = y_true.astype(np.int64).reshape(-1)
        y_pred = np.argmax(probs, axis=1)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(one_hot.shape[0]), y_idx] = 1
        metrics = {
            "macro_f1": float(f1_score(y_idx, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_idx, y_pred, average="micro", zero_division=0)),
            "macro_auroc": _safe_macro_auroc(one_hot, probs),
            "macro_auprc": _safe_macro_auprc(one_hot, probs),
        }
        return metrics

    raise ValueError(f"Unsupported task_type: {task_type}")


def merge_epoch_metrics(metrics_list: list[Dict[str, Any]]) -> Dict[str, float]:
    """Average metrics dictionaries over one epoch."""
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0].keys() if isinstance(metrics_list[0][k], (int, float))]
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(m[key]) for m in metrics_list if key in m]
        if vals:
            out[key] = float(np.mean(vals))
    return out
