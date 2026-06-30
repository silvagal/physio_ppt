"""PTB-XL data preparation and loading for the Strodthoff-compatible protocol."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..constants import (
    CLASS_NAMES,
    OFFICIAL_TEST_FOLD,
    OFFICIAL_TRAIN_FOLDS,
    OFFICIAL_VAL_FOLD,
)
from ..utils.io import ensure_dir


def _zscore_per_lead(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return (signal - mean) / (std + eps)


def _parse_scp_codes(raw: str) -> Dict[str, float]:
    try:
        obj = ast.literal_eval(raw)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _load_diagnostic_map(raw_root: Path) -> Dict[str, str]:
    scp_path = raw_root / "scp_statements.csv"
    scp = pd.read_csv(scp_path, index_col=0)
    if "diagnostic" not in scp.columns or "diagnostic_class" not in scp.columns:
        raise ValueError("scp_statements.csv missing diagnostic columns")
    diag = scp[scp["diagnostic"] == 1.0]["diagnostic_class"].dropna()
    return {str(code): str(cls) for code, cls in diag.to_dict().items()}


def _load_record_wfdb(base_path: Path) -> np.ndarray:
    try:
        import wfdb  # type: ignore
    except Exception as exc:
        raise ImportError(
            "wfdb is required for prepare_ptbxl_strodthoff.py. Install with `pip install wfdb`."
        ) from exc
    signal, _fields = wfdb.rdsamp(str(base_path))
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected record shape {arr.shape} at {base_path}")
    return arr.T


def _resolve_record_base(raw_root: Path, rel_path: str) -> Path:
    candidate = raw_root / rel_path
    if candidate.with_suffix(".hea").exists() or candidate.with_suffix(".dat").exists():
        return candidate
    # Fallback for nested extraction directories.
    rel_name = Path(rel_path).name
    hits = list(raw_root.glob(f"**/{rel_name}.hea"))
    if not hits:
        raise FileNotFoundError(f"Could not locate record `{rel_path}` under `{raw_root}`")
    return hits[0].with_suffix("")


def prepare_ptbxl_strodthoff(
    raw_root: str | Path,
    processed_root: str | Path,
    *,
    fs: int = 100,
    normalize: str = "zscore",
    lead_indices: Optional[Sequence[int]] = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, object]:
    """Prepare PTB-XL with official folds for Strodthoff-compatible downstream."""
    raw = Path(raw_root)
    out = Path(processed_root)
    ensure_dir(out)

    db_path = raw / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    db = pd.read_csv(db_path)
    if "ecg_id" not in db.columns or "strat_fold" not in db.columns:
        raise ValueError("ptbxl_database.csv missing required columns")

    diag_map = _load_diagnostic_map(raw)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    if lead_indices is None:
        lead_indices = list(range(12))
    selected_leads = [int(x) for x in lead_indices]

    rel_col = "filename_lr" if int(fs) == 100 else "filename_hr"
    if rel_col not in db.columns:
        raise ValueError(f"{db_path} does not contain `{rel_col}`")

    split_records: Dict[str, Dict[str, List[np.ndarray]]] = {
        "train": {"signals": [], "labels": [], "ecg_ids": [], "patient_ids": [], "folds": []},
        "val": {"signals": [], "labels": [], "ecg_ids": [], "patient_ids": [], "folds": []},
        "test": {"signals": [], "labels": [], "ecg_ids": [], "patient_ids": [], "folds": []},
    }

    for row in db.itertuples(index=False):
        fold = int(getattr(row, "strat_fold"))
        if fold in OFFICIAL_TRAIN_FOLDS:
            split = "train"
        elif fold == OFFICIAL_VAL_FOLD:
            split = "val"
        elif fold == OFFICIAL_TEST_FOLD:
            split = "test"
        else:
            continue

        rel_path = str(getattr(row, rel_col))
        record_base = _resolve_record_base(raw, rel_path)
        signal = _load_record_wfdb(record_base)
        signal = signal[selected_leads, :]
        if normalize == "zscore":
            signal = _zscore_per_lead(signal)

        y = np.zeros((len(class_names),), dtype=np.float32)
        scp_codes = _parse_scp_codes(str(getattr(row, "scp_codes")))
        for scp_code, weight in scp_codes.items():
            if weight <= 0:
                continue
            super_class = diag_map.get(scp_code)
            if super_class in class_to_idx:
                y[class_to_idx[str(super_class)]] = 1.0

        ecg_id = int(getattr(row, "ecg_id"))
        pid_raw = getattr(row, "patient_id")
        if pd.isna(pid_raw):
            patient_id = -1
        else:
            patient_id = int(float(pid_raw))

        dst = split_records[split]
        dst["signals"].append(signal.astype(np.float32))
        dst["labels"].append(y)
        dst["ecg_ids"].append(np.asarray(ecg_id, dtype=np.int64))
        dst["patient_ids"].append(np.asarray(patient_id, dtype=np.int64))
        dst["folds"].append(np.asarray(fold, dtype=np.int64))

    split_sizes: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        dst = split_records[split]
        if not dst["signals"]:
            raise RuntimeError(f"No samples prepared for split={split}")
        split_dir = ensure_dir(out / split)
        np.savez_compressed(
            split_dir / "records.npz",
            signals=np.stack(dst["signals"]).astype(np.float32),
            labels=np.stack(dst["labels"]).astype(np.float32),
            ecg_ids=np.asarray(dst["ecg_ids"], dtype=np.int64),
            patient_ids=np.asarray(dst["patient_ids"], dtype=np.int64),
            strat_fold=np.asarray(dst["folds"], dtype=np.int64),
            fs=np.asarray([int(fs)], dtype=np.int32),
            lead_indices=np.asarray(selected_leads, dtype=np.int32),
            class_names=np.asarray(class_names),
        )
        split_sizes[split] = len(dst["signals"])

    metadata = {
        "protocol_name": "ptbxl_strodthoff_compatible",
        "task": "superdiagnostic_multilabel",
        "class_names": list(class_names),
        "fs": int(fs),
        "normalize": normalize,
        "lead_indices": selected_leads,
        "official_folds": {"train": list(OFFICIAL_TRAIN_FOLDS), "val": OFFICIAL_VAL_FOLD, "test": OFFICIAL_TEST_FOLD},
        "split_sizes": split_sizes,
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


class PTBXLStrodthoffDataset(Dataset):
    """Dataset over full ECG records for record-level downstream protocol."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        selected_ecg_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.root = Path(root)
        split_path = self.root / split / "records.npz"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        data = np.load(split_path, allow_pickle=True)
        self.signals = data["signals"].astype(np.float32)
        self.labels = data["labels"].astype(np.float32)
        self.ecg_ids = data["ecg_ids"].astype(np.int64)
        self.patient_ids = data["patient_ids"].astype(np.int64)

        if selected_ecg_ids is not None:
            selected = np.asarray(selected_ecg_ids, dtype=np.int64)
            mask = np.isin(self.ecg_ids, selected)
            self.signals = self.signals[mask]
            self.labels = self.labels[mask]
            self.ecg_ids = self.ecg_ids[mask]
            self.patient_ids = self.patient_ids[mask]

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.signals[idx].astype(np.float32)),
            "y": torch.from_numpy(self.labels[idx].astype(np.float32)),
            "ecg_id": torch.tensor(int(self.ecg_ids[idx]), dtype=torch.long),
            "patient_id": torch.tensor(int(self.patient_ids[idx]), dtype=torch.long),
        }


def build_loader(
    root: str | Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    selected_ecg_ids: Optional[Sequence[int]] = None,
) -> DataLoader:
    ds = PTBXLStrodthoffDataset(root=root, split=split, selected_ecg_ids=selected_ecg_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

