"""Split helpers with group-level leakage protection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class SplitIndex:
    """Indices grouped by train/val/test."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def group_train_val_test(
    groups: Sequence[str | int],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> SplitIndex:
    """Create train/val/test index split while keeping group integrity."""
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    groups_arr = np.asarray(groups)
    uniq = np.unique(groups_arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    g_train = set(uniq[:n_train].tolist())
    g_val = set(uniq[n_train : n_train + n_val].tolist())
    g_test = set(uniq[n_train + n_val :].tolist())

    idx = np.arange(groups_arr.shape[0])
    train_idx = idx[np.isin(groups_arr, list(g_train))]
    val_idx = idx[np.isin(groups_arr, list(g_val))]
    test_idx = idx[np.isin(groups_arr, list(g_test))]

    return SplitIndex(train=train_idx, val=val_idx, test=test_idx)


def low_label_group_subsample(
    groups: Sequence[str | int],
    indices: Iterable[int],
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Subsample train indices by selecting a fraction of unique groups."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    idx = np.asarray(list(indices), dtype=np.int64)
    if fraction >= 1.0:
        return idx

    g = np.asarray(groups)
    g_idx = g[idx]
    uniq = np.unique(g_idx)

    n_keep = max(1, int(np.ceil(len(uniq) * fraction)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    keep_groups = set(uniq[:n_keep].tolist())

    keep_mask = np.isin(g_idx, list(keep_groups))
    return idx[keep_mask]
