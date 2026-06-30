"""Basic I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write dictionary as YAML."""
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Read YAML file as dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be dict, got {type(data)}")
    return data


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write JSON with UTF-8 encoding."""
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV without index."""
    out = Path(path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
