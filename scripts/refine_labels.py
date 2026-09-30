"""Silence-aware label refinement (lever L1): report labels + calibrated OOF image predictions.

    python scripts/refine_labels.py --data /kaggle/input/rsna-knee-abnormality-detection \
        --labels labels/llm_states.csv --oof "outputs/r0/oof_fold*.csv" [--oof "outputs/r0b/oof_fold*.csv"] \
        --out labels/refined_r1.csv

Each ``--oof`` is a glob of one run's per-fold files (disjoint rows, concatenated). Several runs
(repeated ``--oof``) are rank-averaged before calibration. Gold studies keep gold labels. ``--targets`` restricts refinement to chosen findings.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneemri.io import read_table  # noqa: E402
from kneemri.constants import ID_COL, TARGETS  # noqa: E402
from kneemri.ensemble import rank_mean  # noqa: E402
from kneemri.labels.refine import explicitness_from_soft, refine_labels  # noqa: E402
from kneemri.labels.teachers import gold_rows, normalise_columns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--oof", required=True, action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-explicit", type=float, default=0.85)
    ap.add_argument("--w-silent", type=float, default=0.25)
    ap.add_argument("--targets", nargs="*", default=None)
    a = ap.parse_args()

    lab = normalise_columns(read_table(a.labels)).set_index(ID_COL)
    y = lab[TARGETS].astype(float)
    ecols = [f"{t}__e" for t in TARGETS]
    explicit = lab[ecols].set_axis(TARGETS, axis=1) if all(c in lab for c in ecols) else explicitness_from_soft(y)
    oofs = []
    for pattern in a.oof:
        files = sorted(glob.glob(pattern)) or [pattern]
        run = pd.concat([read_table(f) for f in files]).dropna(subset=TARGETS).drop_duplicates(ID_COL, keep="last")
        oofs.append(run)
    oof = (oofs[0] if len(oofs) == 1 else rank_mean(oofs)).set_index(ID_COL)[TARGETS]
    gold = gold_rows(read_table(Path(a.data) / "train.csv"))
    refined = refine_labels(y, explicit, oof, a.w_explicit, a.w_silent, gold=gold, targets=a.targets)

    out = refined.copy()
    wcols = [f"{t}__w" for t in TARGETS]
    if all(c in lab for c in wcols):
        # refined cells are now informed by the image too: lift the weight of silent cells
        w = lab[wcols].set_axis(TARGETS, axis=1).clip(lower=0.6)
        for t in TARGETS:
            out[f"{t}__w"] = w[t]
    out.reset_index().to_csv(a.out, index=False)
    changed = (refined - y).abs().mean().round(4)
    print("mean |change| per target:\n", changed.to_string())


if __name__ == "__main__":
    main()
