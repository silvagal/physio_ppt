"""Smoke tests for dataset loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from physio_ppt.data.mitbih import MITBIHBeatsDataset, MITBIHWindowsDataset
from physio_ppt.data.ptbxl import PTBXLRecordsDataset


def _make_synthetic_tree(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "mitbih_windows" / split).mkdir(parents=True, exist_ok=True)
        (root / "mitbih_beats" / split).mkdir(parents=True, exist_ok=True)
        (root / "ptbxl" / split).mkdir(parents=True, exist_ok=True)

        n = 8
        np.savez_compressed(
            root / "mitbih_windows" / split / "windows.npz",
            signals=np.random.randn(n, 2, 1000).astype(np.float32),
            record_ids=np.array([f"r{i}" for i in range(n)]),
            patient_ids=np.array([f"p{i//2}" for i in range(n)]),
            window_start_sample=np.arange(n),
        )
        np.savez_compressed(
            root / "mitbih_beats" / split / "beats.npz",
            signals=np.random.randn(n, 2, 300).astype(np.float32),
            labels=np.random.randint(0, 5, size=(n,), dtype=np.int64),
            record_ids=np.array([f"r{i}" for i in range(n)]),
            patient_ids=np.array([f"p{i//2}" for i in range(n)]),
        )
        np.savez_compressed(
            root / "ptbxl" / split / "records.npz",
            signals=np.random.randn(n, 2, 1000).astype(np.float32),
            labels=np.random.randint(0, 2, size=(n, 5)).astype(np.float32),
            patient_ids=np.array([f"p{i//2}" for i in range(n)]),
            ecg_ids=np.arange(n),
        )


def test_load_small_datasets(tmp_path: Path) -> None:
    _make_synthetic_tree(tmp_path)

    ds_w = MITBIHWindowsDataset(tmp_path / "mitbih_windows", split="train")
    ds_b = MITBIHBeatsDataset(tmp_path / "mitbih_beats", split="train")
    ds_p = PTBXLRecordsDataset(tmp_path / "ptbxl", split="train", crop_size=500)

    assert len(ds_w) == 8
    assert len(ds_b) == 8
    assert len(ds_p) == 8

    xw = ds_w[0]["x"]
    xb = ds_b[0]["x"]
    xp = ds_p[0]["x"]

    assert tuple(xw.shape) == (2, 1000)
    assert tuple(xb.shape) == (2, 300)
    assert tuple(xp.shape) == (2, 500)
