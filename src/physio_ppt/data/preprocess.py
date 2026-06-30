"""Data preparation utilities for MIT-BIH and PTB-XL."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly
from tqdm import tqdm

from .beat_segment import extract_beats, segment_bounds_with_fallback
from .rpeak import detect_rpeaks
from .splits import group_train_val_test
from ..utils.io import ensure_dir


AAMI_MAP = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    "V": 2,
    "E": 2,
    "!": 2,
    "F": 3,
    "f": 3,
    "Q": 4,
    "/": 4,
    "x": 4,
    "|": 4,
    "~": 4,
}


def _zscore_per_lead(sig: np.ndarray) -> np.ndarray:
    mu = np.mean(sig, axis=1, keepdims=True)
    std = np.std(sig, axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (sig - mu) / std


def _resample(sig: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    if fs_orig == fs_target:
        return sig
    return resample_poly(sig, up=fs_target, down=fs_orig, axis=1)


def _list_mitbih_records(raw_dir: Path) -> List[str]:
    records = sorted({p.stem for p in raw_dir.glob("*.hea")})
    if not records:
        raise FileNotFoundError(f"No MIT-BIH .hea files found in: {raw_dir}")
    return records


def _windows(sig: np.ndarray, window_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    chunks: List[np.ndarray] = []
    starts: List[int] = []
    if sig.shape[1] < window_len:
        return np.zeros((0, sig.shape[0], window_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    for s in range(0, sig.shape[1] - window_len + 1, stride):
        chunks.append(sig[:, s : s + window_len].astype(np.float32))
        starts.append(s)
    return np.stack(chunks, axis=0), np.asarray(starts, dtype=np.int64)


def _nearest_ann_label(rpeak: int, ann_samples: np.ndarray, ann_syms: np.ndarray, tol: int) -> int:
    if ann_samples.size == 0:
        return AAMI_MAP["Q"]
    idx = int(np.argmin(np.abs(ann_samples - rpeak)))
    if abs(int(ann_samples[idx]) - int(rpeak)) > tol:
        return AAMI_MAP["Q"]
    sym = str(ann_syms[idx])
    return AAMI_MAP.get(sym, AAMI_MAP["Q"])


def prepare_mitbih(
    raw_dir: str | Path,
    out_dir: str | Path,
    fs_target: int = 500,
    normalize_mode: str = "zscore",
    window_ms: int = 2000,
    window_stride_ms: int = 500,
    beat_pre_ms: int = 200,
    beat_post_ms: int = 400,
    seed: int = 42,
    max_records: int | None = None,
) -> Dict[str, Dict[str, int]]:
    """Prepare MIT-BIH windows and beats with deterministic record-level splits."""
    raw = Path(raw_dir)
    out = Path(out_dir)

    records = _list_mitbih_records(raw)
    if max_records is not None:
        records = records[:max_records]

    split_idx = group_train_val_test(records, train_frac=0.8, val_frac=0.1, seed=seed)
    split_records = {
        "train": [records[i] for i in split_idx.train],
        "val": [records[i] for i in split_idx.val],
        "test": [records[i] for i in split_idx.test],
    }

    window_len = int(round(window_ms * fs_target / 1000.0))
    stride_len = int(round(window_stride_ms * fs_target / 1000.0))

    win_root = out / f"windows_{window_ms}ms_{normalize_mode}"
    beat_root = out / normalize_mode
    for split in ["train", "val", "test"]:
        ensure_dir(win_root / split)
        ensure_dir(beat_root / split)

    summary: Dict[str, Dict[str, int]] = {}
    fallback_count = 0
    total_records = 0
    delineation_attempts = 0
    delineation_failures = 0

    for split, recs in split_records.items():
        win_signals: List[np.ndarray] = []
        win_record_ids: List[str] = []
        win_patient_ids: List[str] = []
        win_starts: List[int] = []

        beat_signals: List[np.ndarray] = []
        beat_labels: List[int] = []
        beat_record_ids: List[str] = []
        beat_patient_ids: List[str] = []

        for rec in tqdm(recs, desc=f"MIT-BIH {split}"):
            total_records += 1
            rec_path = raw / rec

            signal, fields = wfdb.rdsamp(str(rec_path))
            fs_orig = int(round(float(fields["fs"])))
            sig = np.asarray(signal, dtype=np.float32).T
            sig = _resample(sig, fs_orig=fs_orig, fs_target=fs_target)
            if normalize_mode == "zscore":
                sig = _zscore_per_lead(sig)

            w_sig, w_starts = _windows(sig, window_len=window_len, stride=stride_len)
            if w_sig.size > 0:
                win_signals.append(w_sig)
                win_starts.extend(w_starts.tolist())
                win_record_ids.extend([rec] * w_sig.shape[0])
                win_patient_ids.extend([rec] * w_sig.shape[0])

            peaks, info = detect_rpeaks(sig[0], fs=fs_target, prefer_neurokit=True)
            if info.get("fallback") == "true":
                fallback_count += 1

            try:
                ann = wfdb.rdann(str(rec_path), "atr")
                ann_samples = np.round(np.asarray(ann.sample, dtype=np.float64) * fs_target / fs_orig).astype(np.int64)
                ann_syms = np.asarray(ann.symbol)
            except Exception:
                ann_samples = np.zeros((0,), dtype=np.int64)
                ann_syms = np.asarray([], dtype=object)

            beats, kept = extract_beats(sig, peaks, fs=fs_target, pre_ms=beat_pre_ms, post_ms=beat_post_ms)
            if beats.size > 0:
                beat_signals.append(beats)
                tol = int(round(0.05 * fs_target))
                labels = [_nearest_ann_label(r, ann_samples, ann_syms, tol) for r in kept.tolist()]
                beat_labels.extend(labels)
                beat_record_ids.extend([rec] * beats.shape[0])
                beat_patient_ids.extend([rec] * beats.shape[0])

                # Optional delineation check for logging fallback rate.
                r_rel = int(round(beat_pre_ms * fs_target / 1000.0))
                for bi in range(beats.shape[0]):
                    delineation_attempts += 1
                    _, used_fallback = segment_bounds_with_fallback(
                        beat=beats[bi],
                        fs=fs_target,
                        r_index=r_rel,
                        prefer_delineation=True,
                    )
                    if used_fallback:
                        delineation_failures += 1

        if win_signals:
            wins = np.concatenate(win_signals, axis=0)
        else:
            wins = np.zeros((0, 2, window_len), dtype=np.float32)

        np.savez_compressed(
            win_root / split / "windows.npz",
            signals=wins.astype(np.float32),
            record_ids=np.asarray(win_record_ids),
            patient_ids=np.asarray(win_patient_ids),
            window_start_sample=np.asarray(win_starts, dtype=np.int64),
            fs_aligned=np.asarray([fs_target], dtype=np.int32),
            normalize_mode=np.asarray([normalize_mode]),
            window_length_samples=np.asarray([window_len], dtype=np.int32),
        )

        if beat_signals:
            b = np.concatenate(beat_signals, axis=0)
        else:
            b = np.zeros((0, 2, int(round((beat_pre_ms + beat_post_ms) * fs_target / 1000.0))), dtype=np.float32)

        np.savez_compressed(
            beat_root / split / "beats.npz",
            signals=b.astype(np.float32),
            labels=np.asarray(beat_labels, dtype=np.int64),
            record_ids=np.asarray(beat_record_ids),
            patient_ids=np.asarray(beat_patient_ids),
            fs_aligned=np.asarray([fs_target], dtype=np.int32),
            normalize_mode=np.asarray([normalize_mode]),
        )

        summary[split] = {"records": len(recs), "windows": int(wins.shape[0]), "beats": int(b.shape[0])}

    metadata = {
        "fs_target": fs_target,
        "window_ms": window_ms,
        "window_stride_ms": window_stride_ms,
        "normalize_mode": normalize_mode,
        "splits": summary,
        "records": records,
        "rpeak_fallback_records": fallback_count,
        "total_records": total_records,
        "delineation_attempts": delineation_attempts,
        "delineation_failures": delineation_failures,
        "delineation_failure_rate": (
            float(delineation_failures) / float(delineation_attempts)
            if delineation_attempts > 0
            else 0.0
        ),
    }
    with (win_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (beat_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return summary


def _resolve_ptbxl_root(raw_dir: Path) -> Path:
    if (raw_dir / "ptbxl_database.csv").exists():
        return raw_dir
    matches = list(raw_dir.glob("**/ptbxl_database.csv"))
    if not matches:
        raise FileNotFoundError(f"ptbxl_database.csv not found under {raw_dir}")
    return matches[0].parent


def _parse_scp(raw: str) -> Dict[str, float]:
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


def prepare_ptbxl(
    raw_dir: str | Path,
    out_dir: str | Path,
    fs_target: int = 500,
    normalize_mode: str = "zscore",
    seed: int = 42,
    max_records: int | None = None,
) -> Dict[str, int]:
    """Prepare PTB-XL records with patient-level split and superclass labels."""
    root = _resolve_ptbxl_root(Path(raw_dir))
    out = Path(out_dir)

    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag_map = scp[scp["diagnostic"] == 1]["diagnostic_class"].dropna().to_dict()
    classes = sorted(set(str(v) for v in diag_map.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ids = db.index.to_numpy()
    patient_ids = db["patient_id"].to_numpy()
    split_idx = group_train_val_test(patient_ids, train_frac=0.8, val_frac=0.1, seed=seed)

    splits = {
        "train": ids[split_idx.train],
        "val": ids[split_idx.val],
        "test": ids[split_idx.test],
    }

    if max_records is not None:
        for k in splits:
            splits[k] = splits[k][:max_records]

    out_root = out / "superclasses"
    for split in ["train", "val", "test"]:
        ensure_dir(out_root / split)

    split_sizes: Dict[str, int] = {}

    for split, ecg_ids in splits.items():
        sigs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        pids: List[int] = []
        ids_list: List[int] = []
        fs_orig_list: List[int] = []

        for ecg_id in tqdm(ecg_ids.tolist(), desc=f"PTB-XL {split}"):
            row = db.loc[int(ecg_id)]
            path_key = "filename_hr" if isinstance(row.get("filename_hr"), str) else "filename_lr"
            rel = str(row[path_key])
            rec_path = root / rel

            try:
                signal, fields = wfdb.rdsamp(str(rec_path))
            except Exception:
                # Fallback for datasets unpacked in nested folder structures.
                matches = list(root.glob(f"**/{Path(rel).name}.dat"))
                if not matches:
                    continue
                rec_path = matches[0].with_suffix("")
                signal, fields = wfdb.rdsamp(str(rec_path))

            fs_orig = int(round(float(fields["fs"])))
            sig = np.asarray(signal, dtype=np.float32).T
            sig = _resample(sig, fs_orig=fs_orig, fs_target=fs_target)
            if normalize_mode == "zscore":
                sig = _zscore_per_lead(sig)

            y = np.zeros((len(classes),), dtype=np.float32)
            scp_codes = _parse_scp(str(row["scp_codes"]))
            for code, score in scp_codes.items():
                if score <= 0:
                    continue
                super_cls = diag_map.get(code)
                if super_cls is None:
                    continue
                y[class_to_idx[str(super_cls)]] = 1.0

            sigs.append(sig.astype(np.float32))
            labels.append(y)
            pids.append(int(row["patient_id"]))
            ids_list.append(int(ecg_id))
            fs_orig_list.append(int(fs_orig))

        if not sigs:
            raise RuntimeError(f"No PTB-XL samples prepared for split: {split}")

        np.savez_compressed(
            out_root / split / "records.npz",
            signals=np.stack(sigs, axis=0).astype(np.float32),
            labels=np.stack(labels, axis=0).astype(np.float32),
            patient_ids=np.asarray(pids, dtype=np.int64),
            ecg_ids=np.asarray(ids_list, dtype=np.int64),
            fs_orig=np.asarray(fs_orig_list, dtype=np.int32),
            fs_aligned=np.asarray([fs_target], dtype=np.int32),
            normalize_mode=np.asarray([normalize_mode]),
            n_samples=np.asarray([len(sigs)], dtype=np.int32),
            label_classes=np.asarray(classes),
        )
        split_sizes[split] = len(sigs)

    metadata = {
        "fs_aligned": fs_target,
        "normalize_mode": normalize_mode,
        "total_records": int(sum(split_sizes.values())),
        "split_sizes": split_sizes,
        "label_space": "superclasses",
        "classes": classes,
    }
    with (out_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return split_sizes


def prepare_all(data_cfg: Dict[str, object]) -> Dict[str, object]:
    """Prepare datasets according to config dictionary."""
    out: Dict[str, object] = {}

    if bool(data_cfg.get("prepare_mitbih", True)):
        out["mitbih"] = prepare_mitbih(
            raw_dir=str(data_cfg["mitbih_raw_dir"]),
            out_dir=str(data_cfg["mitbih_out_dir"]),
            fs_target=int(data_cfg.get("fs", 500)),
            normalize_mode=str(data_cfg.get("normalize", "zscore")),
            window_ms=int(data_cfg.get("window_ms", 2000)),
            window_stride_ms=int(data_cfg.get("window_stride_ms", 500)),
            beat_pre_ms=int(data_cfg.get("beat_pre_ms", 200)),
            beat_post_ms=int(data_cfg.get("beat_post_ms", 400)),
            seed=int(data_cfg.get("seed", 42)),
            max_records=(
                int(data_cfg["max_mitbih_records"])
                if data_cfg.get("max_mitbih_records") is not None
                else None
            ),
        )

    if bool(data_cfg.get("prepare_ptbxl", True)):
        out["ptbxl"] = prepare_ptbxl(
            raw_dir=str(data_cfg["ptbxl_raw_dir"]),
            out_dir=str(data_cfg["ptbxl_out_dir"]),
            fs_target=int(data_cfg.get("fs", 500)),
            normalize_mode=str(data_cfg.get("normalize", "zscore")),
            seed=int(data_cfg.get("seed", 42)),
            max_records=(
                int(data_cfg["max_ptbxl_records"])
                if data_cfg.get("max_ptbxl_records") is not None
                else None
            ),
        )

    return out
