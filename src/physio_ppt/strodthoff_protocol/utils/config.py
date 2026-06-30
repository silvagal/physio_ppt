"""Config loading wrappers for Strodthoff protocol scripts."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

from physio_ppt.utils.config import apply_overrides, load_config

from .io import save_yaml


def load_protocol_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def save_resolved_config(path: str | Path, cfg: Dict[str, Any]) -> None:
    payload = copy.deepcopy(cfg)
    save_yaml(path, payload)

