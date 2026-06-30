"""Structured logging helpers."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from .io import ensure_dir


def build_logger(name: str = "strodthoff_protocol", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._fp = self.path.open("a", encoding="utf-8")

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        row = {"ts": time.time(), "event": event, **payload}
        self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

