"""Silence-aware label refinement with out-of-fold image predictions (lever L1).

Report labels are *systematically* wrong where the report is silent or vague: radiologists
often leave synovitis, effusion, contusion and OA out of reports that focus on the injury,
while the (image-based) gold labels record them. The public study measured image models
beating every report-label source on exactly those findings.

Round r+1 targets are therefore

    y* = w * y_report + (1 - w) * calibrate(p_image_OOF)

where ``w`` is high where the report is explicit and low where it is silent. ``p_image_OOF``
comes from round-r models that never saw the study (out-of-fold), so a study never teaches
itself. The image prediction is first calibrated onto the report-label scale with a per-target
logistic fit on the explicit rows, because raw OOF scores from a ranking-oriented model are
not probabilities. This is study-level bootstrapping (Reed et al. 2015) / Noisy Student (Xie et
al. 2020), restricted to the cells where the report is least informative.

Gold studies keep their gold labels unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..constants import TARGETS


def explicitness_from_soft(y: pd.DataFrame) -> pd.DataFrame:
    """Fallback explicitness when a teacher has no state column: distance from 0.5, in [0, 1].

    Caution: tables that encode silence as 0 look "explicit negative" under this rule; prefer
    the ``__e`` columns produced by ``llm_labeler.label_reports``.
    """
    return (2 * (y - 0.5)).abs().clip(0, 1)


def calibrate_to_labels(p: np.ndarray, y: np.ndarray, explicit: np.ndarray, min_rows: int = 30) -> np.ndarray:
    """Platt-scale image scores ``p`` onto the probability of an explicit report-positive."""
    eps = 1e-5
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    m = explicit >= 0.99
    yb = (y[m] >= 0.5).astype(int)
    if m.sum() < min_rows or yb.min() == yb.max():
        return p
    lr = LogisticRegression(C=1.0).fit(logit[m], yb)
    return lr.predict_proba(logit)[:, 1]


def refine_labels(
    y_report: pd.DataFrame,
    explicit: pd.DataFrame,
    oof: pd.DataFrame,
    w_explicit: float = 0.85,
    w_silent: float = 0.25,
    gold: pd.DataFrame | None = None,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Blend report labels with calibrated OOF image predictions, per cell.

    ``w = w_silent + (w_explicit - w_silent) * explicitness``. Studies without an OOF prediction
    keep their report label. ``targets`` restricts refinement to a subset of findings (e.g. the
    ones where the image beats the report on gold: Synovitis, Effusion, Contusion, OA).
    """
    targets = targets or TARGETS
    out = y_report.copy()
    common = y_report.index.intersection(oof.index)
    for t in targets:
        y = y_report.loc[common, t].values.astype(float)
        e = explicit.loc[common, t].values.astype(float)
        p = calibrate_to_labels(oof.loc[common, t].values.astype(float), y, e)
        w = w_silent + (w_explicit - w_silent) * np.clip(e, 0, 1)
        out.loc[common, t] = w * y + (1 - w) * p
    if gold is not None:
        g = gold.index.intersection(out.index)
        out.loc[g, TARGETS] = gold.loc[g, TARGETS].values
    return out
