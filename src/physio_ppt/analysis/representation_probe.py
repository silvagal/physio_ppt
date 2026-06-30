"""Optional lightweight representation probe."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def run_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 500,
) -> Tuple[float, Dict[str, float]]:
    """Train a linear probe and return macro-F1."""
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return macro_f1, {"macro_f1": macro_f1}
