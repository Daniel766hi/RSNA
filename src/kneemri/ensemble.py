"""Rank-based fusion. The metric is a per-target ranking, so rank averaging is robust to members
with different calibration and to the host's warning about prevalence shift."""

from __future__ import annotations

import pandas as pd

from .constants import ID_COL, TARGETS


def rank_normalise(df: pd.DataFrame, targets: list[str] = TARGETS) -> pd.DataFrame:
    out = df.copy()
    for t in targets:
        out[t] = df[t].rank(pct=True, method="average")
    return out


def rank_mean(frames: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    """Weighted mean of per-target percentile ranks. Frames are aligned on ``StudyInstanceUID``."""
    weights = weights or [1.0] * len(frames)
    base = frames[0][[ID_COL]].copy()
    acc = pd.DataFrame(0.0, index=base[ID_COL], columns=TARGETS)
    for f, w in zip(frames, weights):
        r = rank_normalise(f).set_index(ID_COL).reindex(base[ID_COL])
        acc += r[TARGETS].fillna(0.5).values * w
    acc /= sum(weights)
    return acc.reset_index()
