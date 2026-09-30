"""Decode every study once into the .npz cache used for training.

    python scripts/cache_volumes.py --data /kaggle/input/rsna-knee-abnormality-detection \
        --split train --out cache/train_384 --img 384 --crop-mm 140 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneemri.io import read_table  # noqa: E402
from kneemri.constants import ID_COL  # noqa: E402
from kneemri.volume import VolumeConfig, build_study, count_slices, save_study  # noqa: E402


def _one(args):
    sid, study_dir, rows, cfg, out = args
    target = Path(out) / f"{sid}.npz"
    if target.exists():
        return sid, "cached"
    try:
        arrays, meta = build_study(study_dir, rows, cfg)
        save_study(target, arrays, meta)
        return sid, f"ok slots={sum(k.endswith('_img') for k in arrays)} side={meta.side}/{meta.side_source}"
    except Exception as exc:  # noqa: BLE001
        return sid, f"FAILED {exc!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--crop-mm", type=float, default=140.0)
    ap.add_argument("--recenter", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    data = Path(a.data)
    series = read_table(data / f"{a.split}_series.csv")
    root = data / f"{a.split}_series"
    series["n_slices"] = count_slices(root, series)
    cfg = VolumeConfig(img=a.img, crop_mm=a.crop_mm, recenter=a.recenter, mirror_right=not a.no_mirror)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(sid, root / sid, g, cfg, a.out) for sid, g in series.groupby(ID_COL)][: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            sid, status = f.result()
            if status.startswith("FAILED") or i % 200 == 0:
                print(f"[{i + 1}/{len(jobs)}] {sid}: {status}", flush=True)


if __name__ == "__main__":
    main()
