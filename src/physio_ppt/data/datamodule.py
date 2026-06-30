"""Data loader assembly for pretraining and fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .mitbih import MITBIHBeatsDataset, MITBIHWindowsDataset
from .ptbxl import PTBXLRecordsDataset
from .splits import low_label_group_subsample


@dataclass
class DataBundle:
    """Train/val/test loaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def _dl(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_pretrain_loaders(cfg: Dict[str, object], seed: int) -> DataBundle:
    """Create pretraining loaders depending on method and data granularity."""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    exp_cfg = cfg["experiment"]

    method = str(exp_cfg["method"])  # ppt_classic | wavepuzzle | physio_ppt | hybrid
    bs = int(train_cfg.get("batch_size", 64))
    nw = int(train_cfg.get("num_workers", 0))
    max_samples = train_cfg.get("max_samples")
    max_samples_int = int(max_samples) if max_samples is not None else None

    if method == "ppt_classic":
        root = str(data_cfg["mitbih_windows_root"])
        ds_train = MITBIHWindowsDataset(root=root, split="train", max_samples=max_samples_int)
        ds_val = MITBIHWindowsDataset(root=root, split="val", max_samples=max_samples_int)
        ds_test = MITBIHWindowsDataset(root=root, split="test", max_samples=max_samples_int)
    else:
        root = str(data_cfg["mitbih_beats_root"])
        ds_train = MITBIHBeatsDataset(root=root, split="train", max_samples=max_samples_int)
        ds_val = MITBIHBeatsDataset(root=root, split="val", max_samples=max_samples_int)
        ds_test = MITBIHBeatsDataset(root=root, split="test", max_samples=max_samples_int)

    return DataBundle(
        train=_dl(ds_train, batch_size=bs, shuffle=True, num_workers=nw),
        val=_dl(ds_val, batch_size=bs, shuffle=False, num_workers=nw),
        test=_dl(ds_test, batch_size=bs, shuffle=False, num_workers=nw),
    )


def build_finetune_loaders(cfg: Dict[str, object], seed: int) -> DataBundle:
    """Create PTB-XL supervised loaders with optional low-label fraction."""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    root = str(data_cfg["ptbxl_root"])
    bs = int(train_cfg.get("batch_size", 32))
    nw = int(train_cfg.get("num_workers", 0))
    crop = train_cfg.get("crop_size")
    crop_size = int(crop) if crop is not None else None
    lead_indices = data_cfg.get("ptbxl_lead_indices")
    lead_idx = [int(x) for x in lead_indices] if lead_indices is not None else None

    ds_train = PTBXLRecordsDataset(
        root=root,
        split="train",
        crop_size=crop_size,
        lead_indices=lead_idx,
        train_mode=True,
        seed=seed,
    )
    ds_val = PTBXLRecordsDataset(
        root=root,
        split="val",
        crop_size=crop_size,
        lead_indices=lead_idx,
        train_mode=False,
        seed=seed,
    )
    ds_test = PTBXLRecordsDataset(
        root=root,
        split="test",
        crop_size=crop_size,
        lead_indices=lead_idx,
        train_mode=False,
        seed=seed,
    )

    frac = float(train_cfg.get("label_fraction", 1.0))
    if frac < 1.0:
        if ds_train.patient_ids is None:
            raise RuntimeError("Low-label sampling requires patient_ids in train split")
        indices = np.arange(len(ds_train))
        keep = low_label_group_subsample(ds_train.patient_ids.astype(str), indices, fraction=frac, seed=seed)
        ds_train = Subset(ds_train, keep.tolist())

    max_train = train_cfg.get("max_train_samples")
    if max_train is not None:
        max_n = int(max_train)
        ds_train = Subset(ds_train, list(range(min(max_n, len(ds_train)))))

    return DataBundle(
        train=_dl(ds_train, batch_size=bs, shuffle=True, num_workers=nw),
        val=_dl(ds_val, batch_size=bs, shuffle=False, num_workers=nw),
        test=_dl(ds_test, batch_size=bs, shuffle=False, num_workers=nw),
    )
