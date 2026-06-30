"""MIT-BIH datasets for window-level and beat-level tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


BatchItem = Dict[str, Union[torch.Tensor, int, str]]


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class MITBIHWindowsDataset(Dataset):
    """Window-level MIT-BIH dataset for PPT-style pretraining."""

    def __init__(self, root: str | Path, split: str = "train", max_samples: Optional[int] = None) -> None:
        self.root = Path(root)
        npz = self.root / split / "windows.npz"
        if not npz.exists():
            raise FileNotFoundError(npz)

        data = np.load(npz, allow_pickle=True)
        self.signals = data["signals"]
        self.record_ids = data.get("record_ids")
        self.patient_ids = data.get("patient_ids")
        self.window_start = data.get("window_start_sample")

        if max_samples is not None:
            self.signals = self.signals[:max_samples]
            if self.record_ids is not None:
                self.record_ids = self.record_ids[:max_samples]
            if self.patient_ids is not None:
                self.patient_ids = self.patient_ids[:max_samples]
            if self.window_start is not None:
                self.window_start = self.window_start[:max_samples]

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> BatchItem:
        x = _to_tensor(self.signals[idx])
        item: BatchItem = {"x": x}
        if self.record_ids is not None:
            item["record_id"] = str(self.record_ids[idx])
        if self.patient_ids is not None:
            item["patient_id"] = str(self.patient_ids[idx])
        if self.window_start is not None:
            item["window_start"] = int(self.window_start[idx])
        return item


class MITBIHBeatsDataset(Dataset):
    """Beat-level MIT-BIH dataset for Physio-PPT and WavePuzzle."""

    def __init__(self, root: str | Path, split: str = "train", max_samples: Optional[int] = None) -> None:
        self.root = Path(root)
        npz = self.root / split / "beats.npz"
        if not npz.exists():
            raise FileNotFoundError(npz)

        data = np.load(npz, allow_pickle=True)
        self.signals = data["signals"]
        self.labels = data.get("labels")
        self.record_ids = data.get("record_ids")
        self.patient_ids = data.get("patient_ids")

        if self.labels is not None and self.labels.dtype.kind in {"U", "S", "O"}:
            uniq = sorted(set(str(x) for x in self.labels.tolist()))
            self.label_to_idx = {u: i for i, u in enumerate(uniq)}
            self.labels = np.asarray([self.label_to_idx[str(x)] for x in self.labels], dtype=np.int64)
        else:
            self.label_to_idx = None

        if max_samples is not None:
            self.signals = self.signals[:max_samples]
            if self.labels is not None:
                self.labels = self.labels[:max_samples]
            if self.record_ids is not None:
                self.record_ids = self.record_ids[:max_samples]
            if self.patient_ids is not None:
                self.patient_ids = self.patient_ids[:max_samples]

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> BatchItem:
        x = _to_tensor(self.signals[idx])
        item: BatchItem = {"x": x}
        if self.labels is not None:
            y = self.labels[idx]
            if np.issubdtype(np.asarray(y).dtype, np.integer):
                item["y"] = torch.tensor(int(y), dtype=torch.long)
            else:
                item["y"] = torch.tensor(float(y), dtype=torch.float32)
        if self.record_ids is not None:
            item["record_id"] = str(self.record_ids[idx])
        if self.patient_ids is not None:
            item["patient_id"] = str(self.patient_ids[idx])
        return item


def mitbih_loader(
    root: str | Path,
    split: str,
    mode: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Factory for MIT-BIH data loaders."""
    if mode == "windows":
        ds = MITBIHWindowsDataset(root=root, split=split, max_samples=max_samples)
    elif mode == "beats":
        ds = MITBIHBeatsDataset(root=root, split=split, max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
