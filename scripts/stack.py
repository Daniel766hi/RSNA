"""Stack the runs of a submission with per-finding LightGBMs, if the pre-registered rule passes.

    python scripts/stack.py --data $DATA --labels labels.csv \
        --pair outputs/r0 raw/raw_g0.csv --pair outputs/r1 raw/raw_g1.csv \
        --baseline submission.csv --out submission.csv

Each ``--pair`` is a run directory (with ``oof_fold*.csv``) and that run's raw test predictions
(``predict.py --raw-dir``). Without a pass, the baseline submission is left untouched. See
``kneemri.stacking`` for the rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneemri.constants import ID_COL, TARGETS  # noqa: E402
from kneemri.io import read_table  # noqa: E402
from kneemri.labels.teachers import gold_rows  # noqa: E402
from kneemri.stacking import baseline_scores, decide, fit_stacker, rank_features  # noqa: E402


def load_oof(run: Path) -> pd.DataFrame:
    parts = [read_table(p) for p in sorted(run.glob("oof_fold*.csv"))]
    if not parts:
        raise FileNotFoundError(f"no oof_fold*.csv in {run}")
    return pd.concat(parts).drop_duplicates(ID_COL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True, help="teacher table used as weak targets")
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("RUN_DIR", "RAW_CSV"))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    train = read_table(Path(a.data) / "train.csv")
    gold_df = gold_rows(train)
    labels = read_table(a.labels).set_index(ID_COL)[TARGETS]
    oofs = [load_oof(Path(r)) for r, _ in a.pair]
    raws = [read_table(p) for _, p in a.pair]
    ids = sorted(set.intersection(*(set(o[ID_COL]) for o in oofs)))
    folds = oofs[0].set_index(ID_COL).reindex(ids)["fold"].to_numpy()
    gold = np.isin(ids, gold_df.index)
    y = labels.reindex(ids).to_numpy(np.float32)
    y[gold] = gold_df.reindex(np.array(ids)[gold]).to_numpy(np.float32)

    base_sub = read_table(a.baseline)
    test_ids = base_sub[ID_COL].tolist()
    x = rank_features(oofs, ids)
    x_test = rank_features(raws, test_ids)
    res = fit_stacker(x, y, x_test, folds, gold)
    weak = ~gold & ~np.isnan(y).any(axis=1)
    rep = decide(y, gold, weak, baseline_scores(x, len(oofs)), res, a.min_gain)
    rep.update(n_train=len(ids), n_gold=int(gold.sum()), n_runs=len(oofs))
    print("[stack]", json.dumps(rep), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    if not rep["use"]:
        print("[stack] rule not met; baseline submission kept", flush=True)
        if Path(a.out) != Path(a.baseline):
            base_sub.to_csv(a.out, index=False)
        return
    stack = pd.DataFrame(res["test"], columns=TARGETS).rank(pct=True).to_numpy()
    base = base_sub[TARGETS].rank(pct=True).to_numpy()
    out = base_sub[[ID_COL]].copy()
    out[TARGETS] = 0.5 * base + 0.5 * stack
    out.to_csv(a.out, index=False)
    print(f"[stack] wrote {a.out}: 50/50 rank blend of baseline and stacker", flush=True)


if __name__ == "__main__":
    main()
