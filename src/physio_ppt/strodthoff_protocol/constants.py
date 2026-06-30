"""Shared constants for the Strodthoff-compatible PTB-XL protocol."""
from __future__ import annotations

from typing import List


DEFAULT_CLASS_ORDER: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = DEFAULT_CLASS_ORDER

OFFICIAL_TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
OFFICIAL_VAL_FOLD = 9
OFFICIAL_TEST_FOLD = 10

