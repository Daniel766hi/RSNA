"""Kaggle submission: DICOM -> predictions for every checkpoint group -> rank-mean -> submission.csv.

    python scripts/predict.py --data /kaggle/input/rsna-knee-abnormality-detection \
        --group /kaggle/input/kneemri-r1/fold*.pt --group /kaggle/input/kneemri-r1b/fold*.pt \
        [--blend-with /kaggle/working/coatnet_family.csv --blend-weight 0.5] --budget-hours 8

Each ``--group`` is one run (its folds are logit-averaged). Groups, and an optional external
submission (e.g. the public CoAtNet family, lever L6), are fused by weighted rank mean.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneemri.io import read_table  # noqa: E402
from kneemri.constants import ID_COL, TARGETS  # noqa: E402
from kneemri.ensemble import rank_mean  # noqa: E402
from kneemri.infer import predict_dicom  # noqa: E402
from kneemri.volume import VolumeConfig, count_slices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--group", action="append", required=True, help="glob of checkpoints of one run")
    ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--crop-mm", type=float, default=140.0)
    ap.add_argument("--recenter", action="store_true")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--blend-with", default=None)
    ap.add_argument("--blend-weight", type=float, default=0.5)
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    t0 = time.time()
    data = Path(a.data)
    test = read_table(data / "test.csv")
    ids = test[ID_COL].astype(str).tolist()
    series = read_table(data / "test_series.csv")
    series["n_slices"] = count_slices(data / "test_series", series)
    vol_cfg = VolumeConfig(img=a.img, crop_mm=a.crop_mm, recenter=a.recenter)

    frames = []
    for gi, pattern in enumerate(a.group):
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            print(f"[predict] no checkpoints for {pattern}; skipped", flush=True)
            continue
        left = a.budget_hours * 3600 - (time.time() - t0)
        budget = left / (len(a.group) - gi)
        print(f"[predict] group {gi}: {len(ckpts)} ckpts, budget {budget / 60:.0f} min", flush=True)
        frames.append(predict_dicom(ids, series, data / "test_series", ckpts, vol_cfg, a.img_size,
                                    num_workers=a.workers, time_budget_s=budget))
    weights = [1.0] * len(frames)
    if a.blend_with:
        ext = read_table(a.blend_with)
        ext[ID_COL] = ext[ID_COL].astype(str)
        frames.append(ext)
        weights.append(a.blend_weight / max(1e-9, 1 - a.blend_weight) * max(1, len(weights)))
    if not frames:
        sub = pd.DataFrame({ID_COL: ids, **{t: 0.5 for t in TARGETS}})
    else:
        base = pd.DataFrame({ID_COL: ids})
        frames = [base.merge(f, on=ID_COL, how="left").fillna(0.5) for f in frames]
        sub = rank_mean(frames, weights)
    sub[[ID_COL] + TARGETS].to_csv(a.out, index=False)
    print(f"[predict] wrote {a.out} ({len(sub)} rows) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
