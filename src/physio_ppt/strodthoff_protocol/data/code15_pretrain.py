"""Scalable CODE-15 loaders for SSL pretraining."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _split_for_exam(exam_id: int, seed: int = 42) -> str:
    """Deterministic split assignment matching Cardio-JEPA convention (70/15/15)."""
    h = int(hashlib.sha256(f"{seed}_{exam_id}".encode("utf-8")).hexdigest()[:8], 16)
    r = h / 0xFFFFFFFF
    if r < 0.70:
        return "train"
    if r < 0.85:
        return "val"
    return "test"


def _zscore_per_lead(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + eps)


def _enforce_shape(x: np.ndarray, num_leads: int, seq_len: int, normalize: bool) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected (C, T), got {x.shape}")
    if x.shape[0] != num_leads and x.shape[1] == num_leads:
        x = x.T
    if x.shape[0] > num_leads:
        x = x[:num_leads, :]
    elif x.shape[0] < num_leads:
        pad = np.zeros((num_leads - x.shape[0], x.shape[1]), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    if x.shape[1] > seq_len:
        x = x[:, :seq_len]
    elif x.shape[1] < seq_len:
        pad = np.zeros((num_leads, seq_len - x.shape[1]), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=1)
    if normalize:
        x = _zscore_per_lead(x)
    return x.astype(np.float32)


def build_code15_hdf5_index(
    raw_root: str | Path,
    index_csv: str | Path,
    *,
    split_seed: int = 42,
    max_records: Optional[int] = None,
) -> Dict[str, int]:
    """Create CSV index over CODE-15 HDF5 shards for lazy loading."""
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("h5py is required to build CODE-15 HDF5 index. Install with `pip install h5py`.") from exc

    raw = Path(raw_root)
    shards = sorted(raw.glob("exams_part*.hdf5"))
    if not shards:
        raise FileNotFoundError(f"No exams_part*.hdf5 found under {raw}")

    rows = []
    total = 0
    for shard in shards:
        with h5py.File(str(shard), "r") as f:
            ds_key = "tracings" if "tracings" in f else "signal"
            if ds_key not in f:
                continue
            n = int(f[ds_key].shape[0])
            exam_ds = f.get("exam_id") or f.get("id_exam")
            if exam_ds is not None:
                exam_ids = np.asarray(exam_ds[:], dtype=np.int64)
            else:
                exam_ids = np.arange(n, dtype=np.int64)
        for local_idx in range(n):
            exam_id = int(exam_ids[local_idx])
            split = _split_for_exam(exam_id, seed=split_seed)
            rows.append(
                {
                    "hdf5_path": str(shard.resolve()),
                    "dataset_key": ds_key,
                    "local_index": int(local_idx),
                    "exam_id": exam_id,
                    "split": split,
                }
            )
            total += 1
            if max_records is not None and total >= int(max_records):
                break
        if max_records is not None and total >= int(max_records):
            break

    if not rows:
        raise RuntimeError("No index rows created for CODE-15")
    df = pd.DataFrame(rows)
    out = Path(index_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return {
        "total_records": int(len(df)),
        "train": int((df["split"] == "train").sum()),
        "val": int((df["split"] == "val").sum()),
        "test": int((df["split"] == "test").sum()),
    }


class CODE15ManifestNPYDataset(Dataset):
    """Lazy per-sample loader for Cardio-JEPA-style processed CODE-15 manifests."""

    def __init__(
        self,
        root: str | Path,
        manifest_csv: str | Path,
        split: str,
        *,
        num_leads: int = 12,
        seq_len: int = 5000,
        normalize: bool = True,
        max_records: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.df = pd.read_csv(manifest_csv)
        if "split" not in self.df.columns or "sample_path" not in self.df.columns:
            raise ValueError("Manifest must contain `split` and `sample_path` columns")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if max_records is not None:
            self.df = self.df.iloc[: int(max_records)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split} in {manifest_csv}")
        self.num_leads = int(num_leads)
        self.seq_len = int(seq_len)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = self.root / str(row["sample_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        x = np.load(path, mmap_mode="r")
        x = np.asarray(x, dtype=np.float32)
        x = _enforce_shape(x, num_leads=self.num_leads, seq_len=self.seq_len, normalize=self.normalize)
        exam_id = int(row.get("exam_id", idx))
        return {"x": torch.from_numpy(x), "exam_id": torch.tensor(exam_id, dtype=torch.long)}


class CODE15HDF5IndexDataset(Dataset):
    """Lazy loader over HDF5 shards using an index CSV."""

    def __init__(
        self,
        index_csv: str | Path,
        split: str,
        *,
        num_leads: int = 12,
        seq_len: int = 4000,
        normalize: bool = True,
        max_records: Optional[int] = None,
    ) -> None:
        self.df = pd.read_csv(index_csv)
        required = {"hdf5_path", "dataset_key", "local_index", "exam_id", "split"}
        missing = required.difference(set(self.df.columns))
        if missing:
            raise ValueError(f"Index CSV missing columns: {sorted(missing)}")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if max_records is not None:
            self.df = self.df.iloc[: int(max_records)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split} in {index_csv}")
        self.num_leads = int(num_leads)
        self.seq_len = int(seq_len)
        self.normalize = bool(normalize)
        self._handles: Dict[str, object] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __len__(self) -> int:
        return int(len(self.df))

    def _get_h5(self, path: str):
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("h5py is required for HDF5 CODE-15 loading. Install with `pip install h5py`.") from exc
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        h5 = self._get_h5(str(row["hdf5_path"]))
        ds = h5[str(row["dataset_key"])]
        x = np.asarray(ds[int(row["local_index"])], dtype=np.float32)
        x = _enforce_shape(x, num_leads=self.num_leads, seq_len=self.seq_len, normalize=self.normalize)
        exam_id = int(row["exam_id"])
        return {"x": torch.from_numpy(x), "exam_id": torch.tensor(exam_id, dtype=torch.long)}


def build_code15_pretrain_loaders(dataset_cfg: Dict[str, object], train_cfg: Dict[str, object]) -> Dict[str, DataLoader]:
    """Create train/val loaders for CODE-15 pretraining."""
    fmt = str(dataset_cfg.get("format", "hdf5_index")).lower()
    split_train = str(dataset_cfg.get("split_train", "train"))
    split_val = str(dataset_cfg.get("split_val", "val"))
    num_leads = int(dataset_cfg.get("num_leads", 12))
    seq_len = int(dataset_cfg.get("seq_len_samples", 4000))
    normalize = bool(dataset_cfg.get("normalize_per_sample", True))
    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 8))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0

    if fmt == "manifest_npy":
        root = dataset_cfg["root"]
        manifest_csv = dataset_cfg["manifest_csv"]
        ds_train = CODE15ManifestNPYDataset(
            root=root,
            manifest_csv=manifest_csv,
            split=split_train,
            num_leads=num_leads,
            seq_len=seq_len,
            normalize=normalize,
            max_records=dataset_cfg.get("max_train_records"),
        )
        ds_val = CODE15ManifestNPYDataset(
            root=root,
            manifest_csv=manifest_csv,
            split=split_val,
            num_leads=num_leads,
            seq_len=seq_len,
            normalize=normalize,
            max_records=dataset_cfg.get("max_val_records"),
        )
    elif fmt == "hdf5_index":
        index_csv = dataset_cfg["index_csv"]
        ds_train = CODE15HDF5IndexDataset(
            index_csv=index_csv,
            split=split_train,
            num_leads=num_leads,
            seq_len=seq_len,
            normalize=normalize,
            max_records=dataset_cfg.get("max_train_records"),
        )
        ds_val = CODE15HDF5IndexDataset(
            index_csv=index_csv,
            split=split_val,
            num_leads=num_leads,
            seq_len=seq_len,
            normalize=normalize,
            max_records=dataset_cfg.get("max_val_records"),
        )
    else:
        raise ValueError(f"Unsupported CODE-15 format: {fmt}")

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return {"train": train_loader, "val": val_loader}

