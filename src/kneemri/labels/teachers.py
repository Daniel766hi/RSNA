"""Report-label *teacher* tables: loading, gold-leak detection, gold agreement and fusion.

A teacher is any table with ``StudyInstanceUID`` plus the 12 target columns holding graded 0-1
labels derived from the report (our LLM labeler, a rule-based extractor, or a public table).

Several public tables copy the ~58 gold labels verbatim. Validating on gold-58 with such a table
silently leaks the answer, so run :func:`detect_gold_leak` on every table before use.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..constants import ID_COL, TARGETS
from ..io import read_table

_ALIASES = {re.sub(r"[^a-z]", "", t.lower()): t for t in TARGETS}
_ALIASES.update({"bakerscyst": "Baker's", "baker": "Baker's", "pfoa": "PF OA", "patellofemoraloa": "PF OA"})


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map loosely named columns (``bakers``, ``Medial_Meniscus``...) onto the canonical targets."""
    ren = {}
    for c in df.columns:
        key = re.sub(r"[^a-z]", "", str(c).lower())
        if key in _ALIASES:
            ren[c] = _ALIASES[key]
        elif key == "studyinstanceuid":
            ren[c] = ID_COL
    return df.rename(columns=ren)


def load_teacher(path: str, id_col: str = ID_COL) -> pd.DataFrame:
    df = normalise_columns(read_table(path).rename(columns={id_col: ID_COL}))
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing target columns {missing}")
    df = df[[ID_COL] + TARGETS].drop_duplicates(ID_COL).set_index(ID_COL).astype(float)
    return df.clip(0.0, 1.0)


def gold_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``train.csv`` that carry all 12 labels, indexed by study."""
    t = train.set_index(ID_COL)
    mask = t[TARGETS].notna().all(axis=1)
    return t.loc[mask, TARGETS].astype(float)


def gold_agreement(teacher: pd.DataFrame, gold: pd.DataFrame) -> pd.Series:
    """Per-target ROC AUC of the teacher's graded labels against gold labels (NaN if undefined)."""
    common = gold.index.intersection(teacher.index)
    out = {}
    for t in TARGETS:
        y = gold.loc[common, t].values
        p = teacher.loc[common, t].values
        out[t] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return pd.Series(out, name="gold_auc")


def detect_gold_leak(teacher: pd.DataFrame, gold: pd.DataFrame, tol: float = 0.98) -> dict:
    """Flag a teacher whose labels on the gold studies reproduce the gold labels (near) exactly.

    Graded report labels essentially never match 58 x 12 binary gold labels exactly after
    rounding, so an exact-match rate >= ``tol`` means the table was built from ``train.csv``'s
    labels.
    """
    common = gold.index.intersection(teacher.index)
    if len(common) == 0:
        return {"n_common": 0, "exact_match": np.nan, "leak": False}
    pred = (teacher.loc[common, TARGETS].values >= 0.5).astype(int)
    exact = float((pred == gold.loc[common, TARGETS].values.astype(int)).mean())
    return {"n_common": int(len(common)), "exact_match": exact, "leak": exact >= tol}


def fuse_teachers(teachers: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    """Weighted mean of graded labels over the teachers that cover each study.

    Probability-space averaging keeps the labels usable as soft BCE targets. The public study
    found that rank-ensembling teachers of shared lineage does not beat the best single clean
    teacher, so prefer diverse teachers (e.g. an LLM teacher + a rule-based one) or just one.
    """
    if not teachers:
        raise ValueError("no teachers")
    weights = weights or [1.0] * len(teachers)
    idx = teachers[0].index
    for t in teachers[1:]:
        idx = idx.union(t.index)
    num = pd.DataFrame(0.0, index=idx, columns=TARGETS)
    den = pd.DataFrame(0.0, index=idx, columns=TARGETS)
    for t, w in zip(teachers, weights):
        t = t.reindex(idx)
        num += t.fillna(0.0) * w
        den += t.notna().astype(float) * w
    return (num / den.replace(0.0, np.nan)).astype(float)
