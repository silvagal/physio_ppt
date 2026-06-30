#!/usr/bin/env python3
"""Alias for evaluate_record_level_strodthoff.py."""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "evaluate_record_level_strodthoff.py"
    runpy.run_path(str(target), run_name="__main__")

