"""Smoke tests for 1-epoch dry-run training."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from physio_ppt.experiments.train_finetune import run_finetune
from physio_ppt.experiments.train_pretrain import run_pretrain


def _make_synthetic_tree(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "mitbih_windows" / split).mkdir(parents=True, exist_ok=True)
        (root / "mitbih_beats" / split).mkdir(parents=True, exist_ok=True)
        (root / "ptbxl" / split).mkdir(parents=True, exist_ok=True)

        n = 16
        np.savez_compressed(
            root / "mitbih_windows" / split / "windows.npz",
            signals=np.random.randn(n, 2, 300).astype(np.float32),
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


def test_dryrun_pretrain_and_finetune(tmp_path: Path) -> None:
    _make_synthetic_tree(tmp_path)

    common_paths = {
        "output_root": str(tmp_path / "outputs"),
        "figures_dir": str(tmp_path / "figures"),
    }

    cfg_pre = {
        "experiment": {"name": "smoke", "method": "physio_ppt"},
        "paths": common_paths,
        "data": {
            "fs": 500,
            "mitbih_windows_root": str(tmp_path / "mitbih_windows"),
            "mitbih_beats_root": str(tmp_path / "mitbih_beats"),
        },
        "model": {
            "backbone": {"name": "resnet1d", "in_channels": 2, "base_channels": 16, "d_model": 64},
            "head": {"proj_dim": 32, "enable_segment_order": True, "num_segment_orders": 6},
        },
        "train": {
            "batch_size": 8,
            "num_workers": 0,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_samples": 16,
        },
        "ssl": {
            "perturb_mode": "mixed",
            "lambda_consistency": 1.0,
            "lambda_contrast": 1.0,
            "lambda_segment": 0.2,
            "use_consistency": True,
            "use_contrastive": True,
        },
    }

    pre = run_pretrain(cfg_pre, seed=42, device="cpu")
    assert Path(pre["checkpoint"]).exists()

    cfg_ft = {
        "experiment": {"name": "smoke_ft", "method": "supervised_strong"},
        "paths": common_paths,
        "data": {
            "ptbxl_root": str(tmp_path / "ptbxl"),
            "ptbxl_lead_indices": [0, 1],
        },
        "model": {
            "backbone": {"name": "resnet1d", "in_channels": 2, "base_channels": 16, "d_model": 64}
        },
        "train": {
            "batch_size": 8,
            "num_workers": 0,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "early_stop_patience": 2,
            "task_type": "multilabel",
            "num_classes": 5,
            "label_fraction": 0.5,
            "crop_size": 500,
            "max_train_samples": 12,
            "pretrained_checkpoint": pre["checkpoint"],
        },
    }

    ft = run_finetune(cfg_ft, seed=42, device="cpu")
    assert Path(ft["checkpoint"]).exists()
    assert Path(ft["predictions"]).exists()
