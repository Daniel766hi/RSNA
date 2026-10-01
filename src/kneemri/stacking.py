"""Second-level stacking: a LightGBM per finding on the out-of-fold predictions of every run.

The image models score each finding separately (apart from the label-interaction layer). A
stacker sees all 12 x G scores at once, so it can use clinical co-occurrence (effusion with
synovitis, contusion with ACL tears, OA across compartments) and weigh the runs per finding.

Guarding against the small gold set:

* Features are per-column percentile ranks. OOF predictions come from one fold model while test
  predictions average all folds, so their scales differ but their ranks are comparable.
* The stacker trains on the weak-labelled studies only (soft targets, cross-entropy objective).
  The gold studies are never used for fitting.
* The decision rule is fixed before seeing the result. Use the stacker only if its gold macro
  AUC beats the rank-mean baseline by ``min_gain`` and its weak-label CV macro AUC is not
  worse. Even then, rank-blend it 50/50 with the baseline instead of replacing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ID_COL, TARGETS
from .metrics import macro_auc


def rank_features(frames: list[pd.DataFrame], ids: list[str]) -> np.ndarray:
    """Stack the per-run, per-target percentile ranks of ``frames`` (aligned on ``ids``)."""
    cols = []
    for f in frames:
        r = f.set_index(ID_COL)[TARGETS].reindex(ids).rank(pct=True)
        cols.append(r.fillna(0.5).to_numpy(np.float32))
    return np.concatenate(cols, axis=1)


def _fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int) -> np.ndarray:
    import lightgbm as lgb  # noqa: PLC0415 - optional dependency (present on Kaggle images)

    m = lgb.LGBMRegressor(objective="cross_entropy", n_estimators=300, learning_rate=0.03, num_leaves=15,
                          min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=1.0, random_state=seed, verbose=-1)
    m.fit(x_tr, y_tr)
    return m.predict(x_te)


def fit_stacker(x: np.ndarray, y: np.ndarray, x_test: np.ndarray, folds: np.ndarray, gold: np.ndarray,
                seed: int = 0) -> dict:
    """Train per-target stackers on the non-gold rows of ``x`` (soft labels ``y``).

    Returns OOF stacker predictions on the weak rows (by ``folds``), predictions on the gold
    rows and on ``x_test`` from a model fit on all weak rows.
    """
    weak = ~gold
    n_t = y.shape[1]
    oof = np.full((len(x), n_t), np.nan, np.float32)
    gold_pred = np.zeros((int(gold.sum()), n_t), np.float32)
    test_pred = np.zeros((len(x_test), n_t), np.float32)
    for t in range(n_t):
        ok = weak & ~np.isnan(y[:, t])
        for k in np.unique(folds[ok]):
            tr, va = ok & (folds != k), ok & (folds == k)
            oof[va, t] = _fit_predict(x[tr], y[tr, t], x[va], seed)
        both = _fit_predict(x[ok], y[ok, t], np.concatenate([x[gold], x_test]), seed)
        gold_pred[:, t], test_pred[:, t] = both[: gold.sum()], both[gold.sum():]
    return {"oof": oof, "gold": gold_pred, "test": test_pred}


def baseline_scores(x: np.ndarray, n_runs: int) -> np.ndarray:
    """The rank-mean ensemble the stacker must beat, from the same rank features."""
    return x.reshape(len(x), n_runs, len(TARGETS)).mean(axis=1)


def decide(y: np.ndarray, gold: np.ndarray, weak_mask: np.ndarray, base: np.ndarray, res: dict,
           min_gain: float = 0.005) -> dict:
    """The decision rule fixed in advance: gold gain >= ``min_gain`` and weak CV not worse."""
    g_base = macro_auc(y[gold], base[gold])
    g_stack = macro_auc(y[gold], res["gold"])
    w_base = macro_auc(y[weak_mask], base[weak_mask])
    w_stack = macro_auc(y[weak_mask], res["oof"][weak_mask])
    use = (g_stack - g_base >= min_gain) and (w_stack >= w_base)
    return {"gold_base": g_base, "gold_stack": g_stack, "weak_base": w_base, "weak_stack": w_stack,
            "use": bool(use)}
