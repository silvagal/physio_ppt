"""PTB-XL exam-level dataset loader with optional cropping."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


BatchItem = Dict[str, Union[torch.Tensor, int, str]]


class PTBXLRecordsDataset(Dataset):
    """PTB-XL records loader (signals + labels) from records.npz."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_size: Optional[int] = None,
        lead_indices: Optional[list[int]] = None,
        train_mode: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        npz = self.root / split / "records.npz"
        if not npz.exists():
            raise FileNotFoundError(npz)

        data = np.load(npz, allow_pickle=True)
        self.signals = data["signals"].astype(np.float32)
        self.labels = data.get("labels")
        self.patient_ids = data.get("patient_ids")
        self.ecg_ids = data.get("ecg_ids")

        self.crop_size = crop_size
        self.lead_indices = lead_indices
        self.train_mode = train_mode
        self.seed = int(seed)

        if max_samples is not None:
            self.signals = self.signals[:max_samples]
            if self.labels is not None:
                self.labels = self.labels[:max_samples]
            if self.patient_ids is not None:
                self.patient_ids = self.patient_ids[:max_samples]
            if self.ecg_ids is not None:
                self.ecg_ids = self.ecg_ids[:max_samples]

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def _crop(self, x: np.ndarray, idx: int) -> np.ndarray:
        if self.crop_size is None or x.shape[-1] <= self.crop_size:
            return x

        if self.train_mode:
            rng = np.random.default_rng(self.seed + idx)
            start = int(rng.integers(0, x.shape[-1] - self.crop_size + 1))
        else:
            start = int((x.shape[-1] - self.crop_size) // 2)
        return x[:, start : start + self.crop_size]

    def __getitem__(self, idx: int) -> BatchItem:
        x_np = self._crop(self.signals[idx], idx)
        if self.lead_indices is not None:
            x_np = x_np[self.lead_indices, :]
        x = torch.from_numpy(x_np.astype(np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)

        item: BatchItem = {"x": x}

        if self.labels is not None:
            y = self.labels[idx]
            y_arr = np.asarray(y)
            if y_arr.ndim == 0:
                item["y"] = torch.tensor(int(y_arr), dtype=torch.long)
            else:
                item["y"] = torch.from_numpy(y_arr.astype(np.float32))

        if self.patient_ids is not None:
            item["patient_id"] = str(self.patient_ids[idx])
        if self.ecg_ids is not None:
            item["ecg_id"] = str(self.ecg_ids[idx])

        return item


def ptbxl_loader(
    root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    crop_size: Optional[int] = None,
    lead_indices: Optional[list[int]] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """Create PTB-XL DataLoader."""
    ds = PTBXLRecordsDataset(
        root=root,
        split=split,
        crop_size=crop_size,
        lead_indices=lead_indices,
        train_mode=shuffle,
        max_samples=max_samples,
        seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
