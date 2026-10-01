"""Stage 2: retrain each fold's MIL head on full-study context (see ``kneemri.head``).

    python scripts/train_head.py --data $DATA --cache cache_320 --labels labels.csv \
        --run outputs/r1 --out outputs/r1h --folds 0 2 4 --views 3 --epochs 30

Writes ``fold<k>.pt`` (stage-1 backbone + new head), ``oof_fold<k>.csv`` and
``metrics_f<folds>.json`` in the same formats as ``train_cv.py``. The run therefore drops into
``predict.py``, ``refine_labels.py`` and the submission stage unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneemri.constants import ID_COL, TARGETS  # noqa: E402
from kneemri.head import HeadConfig, extract_features, predict_head, train_head  # noqa: E402
from kneemri.io import read_table  # noqa: E402
from kneemri.labels.teachers import normalise_columns  # noqa: E402
from kneemri.metrics import macro_auc, per_target_auc  # noqa: E402
from kneemri.model import KneeMIL, ModelConfig  # noqa: E402
from kneemri.series import DEFAULT_SLOTS  # noqa: E402
from kneemri.train import build_targets, make_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True, help="cache dir(s), comma-separated")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--run", required=True, help="stage-1 run dir with fold<k>.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=1, help="heads per fold, averaged in logit space via OOF mean")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N studies")
    a = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (run / "volume.json").exists():
        shutil.copy(run / "volume.json", out / "volume.json")
    ckpts = {int(p.stem[4:]): p for p in run.glob("fold*.pt") if p.stem[4:].isdigit()}
    folds_to_run = sorted(ckpts) if a.folds is None else [k for k in a.folds if k in ckpts]
    first = torch.load(ckpts[folds_to_run[0]], map_location="cpu", weights_only=False)
    tcfg = first["train_cfg"]

    train = read_table(Path(a.data) / "train.csv")
    if a.limit:
        train = train.head(a.limit)
    lab = normalise_columns(read_table(a.labels)).set_index(ID_COL)
    labels = lab[TARGETS].astype(float)
    wcols = [f"{t}__w" for t in TARGETS]
    weights = lab[wcols].set_axis(TARGETS, axis=1).astype(float) if all(c in lab for c in wcols) else None
    cache_dirs = [Path(c) for c in a.cache.split(",") if c]
    found = train[ID_COL].map(lambda s: next((d / f"{s}.npz" for d in cache_dirs if (d / f"{s}.npz").exists()), None))
    train = train[found.notna()].reset_index(drop=True)
    paths = [p for p in found if p is not None]
    y, w, gold = build_targets(train, labels, weights, tcfg.get("gold_weight", 3.0))
    folds = make_folds(train, tcfg.get("n_folds", 5), tcfg.get("seed", 42)).reindex(train[ID_COL]).values
    hcfg = HeadConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, auc_weight=tcfg.get("auc_weight", 0.1))
    print(f"[head] {len(train)} studies, folds {folds_to_run}, views {a.views}, device {device}", flush=True)

    oof = np.full((len(train), len(TARGETS)), np.nan, np.float32)
    for k in folds_to_run:
        state = torch.load(ckpts[k], map_location="cpu", weights_only=False)
        mcfg = ModelConfig(**{**state["model_cfg"], "pretrained": False})
        model = KneeMIL(mcfg)
        model.load_state_dict(state["state_dict"])
        model.to(device)
        t1 = time.time()
        feats = extract_features(model, paths, DEFAULT_SLOTS[: mcfg.n_slots], state["train_cfg"].get("img_size"),
                                 a.views, device, a.workers, seed=k)
        print(f"[head] fold {k}: features in {(time.time() - t1) / 60:.1f} min", flush=True)
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        preds = []
        for s in range(a.seeds):  # every seed is saved, so predict.py averages the same heads as the OOF
            hcfg.seed = 1000 * k + s
            train_head(model, feats, y, w, tr, hcfg, device)
            preds.append(predict_head(model, feats, va, device))
            torch.save({"model_cfg": state["model_cfg"],
                        "train_cfg": {**state["train_cfg"], "stage2_head": dict(vars(hcfg))},
                        "state_dict": {n: v.detach().cpu() for n, v in model.state_dict().items()}},
                       out / (f"fold{k}.pt" if s == 0 else f"fold{k}s{s}.pt"))
        pred = np.mean(preds, axis=0)
        oof[va] = pred
        part = pd.DataFrame(pred, columns=TARGETS)
        part.insert(0, ID_COL, train[ID_COL].values[va])
        part.insert(1, "fold", k)
        part.to_csv(out / f"oof_fold{k}.csv", index=False)
        g = gold[va]
        print(f"[head] fold {k}: done in {(time.time() - t1) / 60:.1f} min; "
              f"gold macro {macro_auc(y[va][g], pred[g]) if g.any() else float('nan'):.4f}", flush=True)
        del feats

    done = ~np.isnan(oof).any(1)
    weak = done & ~gold
    metrics = {
        "weak_macro": macro_auc(y[weak], oof[weak]) if weak.any() else None,
        "gold_macro": macro_auc(y[done & gold], oof[done & gold]) if (done & gold).any() else None,
        "gold_per_target": per_target_auc(y[done & gold], oof[done & gold]) if (done & gold).any() else None,
        "n_weak": int(weak.sum()), "n_gold": int((done & gold).sum()),
    }
    suffix = "_f" + "".join(map(str, folds_to_run))
    (out / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    print(f"[head] total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
