"""Macro AUC (the competition metric) and paired bootstrap comparisons for small gold sets."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .constants import TARGETS


def per_target_auc(y: np.ndarray, p: np.ndarray, targets: list[str] = TARGETS) -> dict[str, float]:
    """AUC per column; soft labels are binarised at 0.5; undefined columns give NaN."""
    out = {}
    for k, t in enumerate(targets):
        yk = y[:, k]
        ok = ~np.isnan(yk)
        yb = (yk[ok] >= 0.5).astype(int)
        out[t] = float(roc_auc_score(yb, p[ok, k])) if 0 < yb.sum() < len(yb) else float("nan")
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    vals = [v for v in per_target_auc(y, p).values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the macro-AUC difference B - A over studies (resampled jointly).

    With ~58 gold studies the macro-AUC standard error is ~0.02-0.03, so use this as a
    regression guard (does B clearly lose?) rather than to rank close candidates.
    """
    rng = np.random.default_rng(seed)
    m = len(y)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        a, b = macro_auc(y[idx], p_a[idx]), macro_auc(y[idx], p_b[idx])
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(b - a)
    d = np.asarray(diffs)
    return {
        "delta": macro_auc(y, p_b) - macro_auc(y, p_a),
        "ci90": (float(np.percentile(d, 5)), float(np.percentile(d, 95))),
        "p_gt_0": float((d > 0).mean()),
    }
