"""Cross-validated training on the ``.npz`` cache with soft weak labels and a gold-58 guard.

Validation-leak guards:

* Folds are grouped by a hash of the normalised report text. Studies sharing a report (and
  therefore a derived label) stay in one fold.
* Gold studies are spread evenly over folds and scored only out-of-fold.

Checkpointing: the EMA weights after the last epoch are saved. With ~12 gold studies per fold,
per-epoch selection on gold would be selection on noise.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .constants import ID_COL, TARGETS
from .data import KneeStudyDataset, collate
from .losses import composite_loss
from .metrics import macro_auc, per_target_auc
from .model import KneeMIL, ModelConfig, param_groups
from .series import DEFAULT_SLOTS


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 4
    lr_backbone: float = 4e-5
    lr_head: float = 4e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    ema_decay: float = 0.998
    auc_weight: float = 0.1
    gold_weight: float = 3.0
    n_folds: int = 5
    folds: list[int] | None = None          # subset of folds to run (e.g. [0] for a quick A/B)
    seed: int = 42
    num_workers: int = 4
    amp: bool = True
    img_size: int | None = None             # resize cached windows at load time
    window_dropout: float = 0.2
    slot_dropout: float = 0.1
    max_windows: int | None = 16            # random windows per study per training step (speed)
    grad_accum: int = 1
    out_dir: str = "outputs/run"
    model: ModelConfig = field(default_factory=ModelConfig)


# --------------------------------------------------------------------------------------------
# folds and targets


def _report_hash(text) -> str:
    norm = " ".join(str(text if isinstance(text, str) else "").lower().split())
    return hashlib.md5(norm.encode()).hexdigest()


def make_folds(train: pd.DataFrame, n_folds: int, seed: int = 42) -> pd.Series:
    """Fold per study: report-hash groups assigned greedily to the smallest fold, gold first."""
    df = train[[ID_COL]].copy()
    df["g"] = train["Report"].map(_report_hash) if "Report" in train else train[ID_COL]
    df["gold"] = train[TARGETS].notna().all(axis=1).values if all(t in train for t in TARGETS) else False
    groups = df.groupby("g").agg(n=(ID_COL, "size"), gold=("gold", "sum")).reset_index()
    rng = np.random.default_rng(seed)
    groups = groups.iloc[rng.permutation(len(groups))]
    sizes, golds = np.zeros(n_folds), np.zeros(n_folds)
    fold_of = {}
    for _, g in groups[groups["gold"] > 0].iterrows():
        k = int(np.argmin(golds * 1e6 + sizes))
        fold_of[g["g"]] = k
        golds[k] += g["gold"]
        sizes[k] += g["n"]
    for _, g in groups[groups["gold"] == 0].sort_values("n", ascending=False, kind="stable").iterrows():
        k = int(np.argmin(sizes))
        fold_of[g["g"]] = k
        sizes[k] += g["n"]
    return pd.Series(df["g"].map(fold_of).values, index=df[ID_COL].values, name="fold")


def build_targets(
    train: pd.DataFrame, labels: pd.DataFrame, weights: pd.DataFrame | None = None, gold_weight: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Y [N,12], W [N,12], is_gold [N]) in ``train`` order.

    Gold rows use gold labels at ``gold_weight``; other rows use the (soft) weak labels with
    optional per-cell weights; cells without any label get weight 0.
    """
    ids = train[ID_COL].values
    gold_mask = train[TARGETS].notna().all(axis=1).values if all(t in train for t in TARGETS) else np.zeros(len(train), bool)
    lab = labels.reindex(ids)[TARGETS].values.astype(np.float32)
    wt = np.ones_like(lab) if weights is None else weights.reindex(ids)[TARGETS].fillna(0).values.astype(np.float32)
    y = lab.copy()
    w = np.where(np.isnan(lab), 0.0, wt).astype(np.float32)
    if gold_mask.any():
        y[gold_mask] = train.loc[gold_mask, TARGETS].values.astype(np.float32)
        w[gold_mask] = gold_weight
    return y, w, gold_mask


# --------------------------------------------------------------------------------------------
# training


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self, like: torch.nn.Module) -> dict:
        ref = like.state_dict()
        return {k: v.to(ref[k].dtype) for k, v in self.shadow.items()}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_loader(model: KneeMIL, loader: DataLoader, device: torch.device, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for b in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(b["x"].to(device), b["slot"].to(device), b["pos"].to(device), b["mask"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)))


def train_fold(
    fold: int, paths: list[Path], y: np.ndarray, w: np.ndarray, folds: np.ndarray, cfg: TrainConfig, device: torch.device
) -> tuple[np.ndarray, np.ndarray, Path]:
    seed_all(cfg.seed + fold)
    tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    ds_kw = dict(slots=DEFAULT_SLOTS[: cfg.model.n_slots], img_size=cfg.img_size)
    tr_ds = KneeStudyDataset([paths[i] for i in tr], y[tr], w[tr], train=True, window_dropout=cfg.window_dropout,
                             slot_dropout=cfg.slot_dropout, max_windows=cfg.max_windows, seed=cfg.seed + fold, **ds_kw)
    va_ds = KneeStudyDataset([paths[i] for i in va], y[va], w[va], train=False, **ds_kw)
    tr_dl = DataLoader(tr_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate,
                       drop_last=True, persistent_workers=cfg.num_workers > 0)
    # eval windows are not capped by max_windows, so keep eval batches small
    va_dl = DataLoader(va_ds, max(1, cfg.batch_size // 2), shuffle=False, num_workers=cfg.num_workers, collate_fn=collate)

    model = KneeMIL(cfg.model).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))
    steps = max(1, cfg.epochs * len(tr_dl) // cfg.grad_accum)
    warm = max(1, int(cfg.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for ep in range(cfg.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for it, b in enumerate(tr_dl):
            with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
                logits = model(b["x"].to(device), b["slot"].to(device), b["pos"].to(device), b["mask"].to(device))
            loss = composite_loss(logits.float(), b["y"].to(device), b["w"].to(device), cfg.auc_weight) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)
            run += loss.item() * cfg.grad_accum
        log.append({"fold": fold, "epoch": ep, "loss": run / max(1, len(tr_dl)), "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)

    model.load_state_dict(ema.state_dict(model))
    pred = predict_loader(model, va_dl, device, cfg.amp)
    ckpt = out_dir / f"fold{fold}.pt"
    torch.save({"model_cfg": cfg.model.to_dict(), "train_cfg": _jsonable(cfg), "state_dict": model.state_dict()}, ckpt)
    (out_dir / f"fold{fold}_log.json").write_text(json.dumps(log, indent=1))
    return va, pred, ckpt


def _jsonable(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    return json.loads(json.dumps(d, default=str))


def run_cv(
    train: pd.DataFrame, cache_dir: str | Path, labels: pd.DataFrame, weights: pd.DataFrame | None, cfg: TrainConfig
) -> pd.DataFrame:
    """Train every requested fold; write ``oof.csv`` and ``metrics.json`` into ``cfg.out_dir``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # a cache may be split over several directories (comma-separated), e.g. Kaggle output shards
    cache_dirs = [Path(c) for c in str(cache_dir).split(",") if c]

    def _path(sid: str) -> Path | None:
        return next((d / f"{sid}.npz" for d in cache_dirs if (d / f"{sid}.npz").exists()), None)

    found = train[ID_COL].map(_path)
    train = train[found.notna()].reset_index(drop=True)
    paths = [p for p in found if p is not None]
    y, w, gold = build_targets(train, labels, weights, cfg.gold_weight)
    folds = make_folds(train, cfg.n_folds, cfg.seed).reindex(train[ID_COL]).values
    oof = np.full((len(train), len(TARGETS)), np.nan, np.float32)
    out_dir = Path(cfg.out_dir)
    run_folds = list(cfg.folds) if cfg.folds is not None else list(range(cfg.n_folds))
    for k in run_folds:
        idx, pred, _ = train_fold(k, paths, y, w, folds, cfg, device)
        oof[idx] = pred
        # one file per fold, so folds trained by parallel processes (one per GPU) never collide
        part = pd.DataFrame(pred, columns=TARGETS)
        part.insert(0, ID_COL, train[ID_COL].values[idx])
        part.insert(1, "fold", k)
        part.to_csv(out_dir / f"oof_fold{k}.csv", index=False)

    out = pd.DataFrame(oof, columns=TARGETS)
    out.insert(0, ID_COL, train[ID_COL].values)
    out.insert(1, "fold", folds)
    suffix = "" if cfg.folds is None else "_f" + "".join(map(str, run_folds))
    out.to_csv(out_dir / f"oof{suffix}.csv", index=False)

    done = ~np.isnan(oof).any(1)
    weak = done & ~gold
    metrics = {
        "weak_macro": macro_auc(y[weak], oof[weak]) if weak.any() else None,
        "gold_macro": macro_auc(y[done & gold], oof[done & gold]) if (done & gold).any() else None,
        "gold_per_target": per_target_auc(y[done & gold], oof[done & gold]) if (done & gold).any() else None,
        "n_weak": int(weak.sum()),
        "n_gold": int((done & gold).sum()),
    }
    (out_dir / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    return out
