"""YAML config loading and merging utilities."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from .io import load_yaml


ConfigDict = Dict[str, Any]


def deep_update(base: ConfigDict, patch: ConfigDict) -> ConfigDict:
    """Recursively merge two dictionaries.

    The input dictionaries are not mutated.
    """
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_recursive(path: Path, seen: Set[Path]) -> ConfigDict:
    if path in seen:
        raise RuntimeError(f"Circular config reference detected at: {path}")
    seen.add(path)

    cfg = load_yaml(path)
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: ConfigDict = {}
    for base_rel in bases:
        base_path = (path.parent / base_rel).resolve()
        merged = deep_update(merged, _resolve_recursive(base_path, seen))

    merged = deep_update(merged, cfg)
    merged["_config_path"] = str(path)
    return merged


def load_config(config_path: str | Path) -> ConfigDict:
    """Load YAML config and recursively merge `_base_` files."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return _resolve_recursive(path, seen=set())


def apply_overrides(config: ConfigDict, overrides: Iterable[str]) -> ConfigDict:
    """Apply CLI overrides in dot-list format.

    Example
    -------
    `train.batch_size=64`, `model.name=resnet1d`
    """
    out = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override `{item}`. Expected key=value")
        key, raw = item.split("=", 1)
        parts = key.split(".")

        value: Any
        raw_l = raw.lower()
        if raw_l in {"true", "false"}:
            value = raw_l == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw

        node = out
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return out


def config_hash(config: ConfigDict, length: int = 10) -> str:
    """Generate stable short hash for config snapshots."""
    if length <= 0:
        raise ValueError("length must be > 0")
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]
