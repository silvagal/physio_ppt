"""Low-label split generation for official PTB-XL train folds."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ..utils.io import ensure_dir, save_json


def _fraction_tag(fraction: float) -> str:
    pct = int(round(fraction * 100))
    return f"{pct:02d}pct"


def generate_low_label_split(
    train_ecg_ids: Sequence[int],
    fraction: float,
    seed: int,
    out_dir: str | Path,
    *,
    method: str = "record_random",
) -> Dict[str, object]:
    """Generate and persist one low-label split for reproducible runs.

    Notes
    -----
    This implementation samples uniformly at record level. Multilabel iterative
    stratification is intentionally not applied by default to avoid additional
    external dependencies and instability at 1% in sparse labels.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    ecg_ids = np.asarray(train_ecg_ids, dtype=np.int64)
    if ecg_ids.size == 0:
        raise ValueError("train_ecg_ids is empty")

    n_keep = max(1, int(math.ceil(float(fraction) * float(ecg_ids.size))))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(ecg_ids.size)
    keep_ids = np.sort(ecg_ids[perm[:n_keep]])

    out_root = ensure_dir(out_dir)
    out_path = out_root / f"seed_{seed}" / f"fraction_{_fraction_tag(fraction)}.json"
    payload = {
        "seed": int(seed),
        "fraction": float(fraction),
        "method": str(method),
        "num_train_records_full": int(ecg_ids.size),
        "num_train_records_selected": int(keep_ids.size),
        "selected_ecg_ids": keep_ids.astype(int).tolist(),
    }
    save_json(out_path, payload)
    return {"split_file": str(out_path), **payload}

