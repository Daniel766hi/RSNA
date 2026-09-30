"""Cross-validated training.

    python scripts/train_cv.py --data /kaggle/input/rsna-knee-abnormality-detection \
        --cache cache/train_384 --labels labels/teacher.csv --out outputs/r0_convnext_s \
        --backbone convnext_small.fb_in22k_ft_in1k_384 --epochs 12 --batch-size 4

``--labels`` is a table with StudyInstanceUID + 12 soft-label columns (a teacher table,
``label_reports_llm.py`` output, or ``refine_labels.py`` output). If it also holds
``<target>__w`` columns, they are used as per-cell sample weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneemri.io import read_table  # noqa: E402
from kneemri.constants import ID_COL, TARGETS  # noqa: E402
from kneemri.labels.teachers import normalise_columns  # noqa: E402
from kneemri.model import ModelConfig  # noqa: E402
from kneemri.train import TrainConfig, run_cv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k_384")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-backbone", type=float, default=4e-5)
    ap.add_argument("--lr-head", type=float, default=4e-4)
    ap.add_argument("--auc-weight", type=float, default=0.1)
    ap.add_argument("--gold-weight", type=float, default=3.0)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=16)
    ap.add_argument("--no-position", action="store_true", help="ablation: permutation-invariant pool (L2 off)")
    ap.add_argument("--ctx-layers", type=int, default=1)
    ap.add_argument("--label-layers", type=int, default=1, help="0 = no label interaction (L3 off)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="subset of studies for quick A/B runs")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    train = read_table(Path(a.data) / "train.csv")
    if a.limit:
        gold = train[TARGETS].notna().all(axis=1)
        train = pd.concat([train[gold], train[~gold].sample(a.limit, random_state=a.seed)])
    lab = normalise_columns(read_table(a.labels)).set_index(ID_COL)
    labels = lab[TARGETS].astype(float)
    wcols = [f"{t}__w" for t in TARGETS]
    weights = lab[wcols].set_axis(TARGETS, axis=1).astype(float) if all(c in lab for c in wcols) else None

    cfg = TrainConfig(
        epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum, lr_backbone=a.lr_backbone,
        lr_head=a.lr_head, auc_weight=a.auc_weight, gold_weight=a.gold_weight, n_folds=a.n_folds,
        folds=a.folds, img_size=a.img_size, max_windows=a.max_windows, num_workers=a.workers,
        seed=a.seed, out_dir=a.out,
        model=ModelConfig(backbone=a.backbone, pretrained=not a.no_pretrained, use_position=not a.no_position,
                          ctx_layers=a.ctx_layers, label_layers=a.label_layers, grad_ckpt=a.grad_ckpt),
    )
    run_cv(train, a.cache, labels, weights, cfg)


if __name__ == "__main__":
    main()
