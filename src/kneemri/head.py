"""Stage 2: retrain the MIL head on full-study context over frozen, cached backbone features.

Stage 1 (``train.py``) fine-tunes the backbone on 16 random windows per study per step, because
a backbone forward over all windows does not fit a T4. The context transformer, the finding
queries and the label-interaction layer therefore learn from ~28% of a study (16 of up to 58
windows), but at inference they see every window. Stage 2 removes that train/test mismatch:

1. With each fold's stage-1 backbone frozen, encode every window of every study once. View 0
   is clean; further views are study-consistent augmentations of the same windows.
2. Train a fresh head on those features with all windows present (light feature-level window
   dropout as regularisation). This takes minutes per fold, so many epochs and seeds are cheap.
3. Save a regular ``KneeMIL`` checkpoint (stage-1 backbone + new head). ``predict.py`` and the
   submission stage use it unchanged.

Out-of-fold hygiene: the fold-k backbone never trained on fold-k studies, and the fold-k head
trains only on the other folds. The fold-k OOF predictions stay out of sample.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import KneeStudyDataset
from .losses import composite_loss
from .model import KneeMIL


@dataclass
class HeadConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    feat_dropout: float = 0.15      # drop whole windows from the cached features
    auc_weight: float = 0.1
    ema_decay: float = 0.99
    seed: int = 0
    reinit: bool = True             # fresh head (True) or continue from the stage-1 head


@torch.no_grad()
def extract_features(model: KneeMIL, paths: list, slots, img_size: int | None, views: int,
                     device: torch.device, num_workers: int = 2, chunk: int = 64, seed: int = 0) -> list[dict]:
    """Per study: ``{"f": fp16 [V,W,F], "slot": [W], "pos": [W]}``, all windows, ``views`` views."""
    model.eval()
    out: list[dict] = [{} for _ in paths]
    for v in range(views):
        ds = KneeStudyDataset(paths, slots=slots, img_size=img_size, train=v > 0, window_dropout=0.0,
                              slot_dropout=0.0, max_windows=None, seed=seed + 1000 * v)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                        collate_fn=lambda b: b[0])
        for i, item in enumerate(dl):
            x = item["x"]
            if x.shape[0] == 0:
                f = torch.zeros(0, model.backbone.num_features, dtype=torch.float16)
            else:
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    f = torch.cat([model.backbone(x[j: j + chunk].to(device)) for j in range(0, len(x), chunk)])
                f = f.half().cpu()
            if v == 0:
                out[i] = {"f": [f], "slot": item["slot"], "pos": item["pos"]}
            else:
                out[i]["f"].append(f)
    for o in out:
        o["f"] = torch.stack(o["f"])
    return out


def _batch(feats: list[dict], idx: list[int], rng: random.Random, train: bool, drop: float):
    rows = []
    for i in idx:
        s = feats[i]
        v = rng.randrange(s["f"].shape[0]) if train else 0
        f, slot, pos = s["f"][v], s["slot"], s["pos"]
        if train and drop > 0 and len(f) > 1:
            keep = torch.tensor([rng.random() >= drop for _ in range(len(f))])
            if keep.any():
                f, slot, pos = f[keep], slot[keep], pos[keep]
        rows.append((f, slot, pos))
    w_max = max(1, max(len(r[0]) for r in rows))
    dim = feats[idx[0]]["f"].shape[-1]
    x = torch.zeros(len(rows), w_max, dim)
    sl = torch.zeros(len(rows), w_max, dtype=torch.long)
    po = torch.zeros(len(rows), w_max)
    mask = torch.zeros(len(rows), w_max, dtype=torch.bool)
    for j, (f, slot, pos) in enumerate(rows):
        n = len(f)
        x[j, :n], sl[j, :n], po[j, :n], mask[j, :n] = f.float(), slot, pos, True
    return x, sl, po, mask


def _reset_head(model: KneeMIL, seed: int) -> None:
    """Re-initialise every non-backbone module (fresh head, same architecture)."""
    torch.manual_seed(seed)
    backbone = set(model.backbone.modules())
    for m in model.modules():
        if m is model or m in backbone:
            continue
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif hasattr(m, "_reset_parameters"):  # nn.MultiheadAttention
            m._reset_parameters()
    with torch.no_grad():
        model.queries.normal_(0, 0.02)
        model.head_w.normal_(0, 0.02)
        model.head_b.zero_()


def train_head(model: KneeMIL, feats: list[dict], y: np.ndarray, w: np.ndarray, tr: np.ndarray,
               cfg: HeadConfig, device: torch.device) -> KneeMIL:
    """Train every non-backbone parameter of ``model`` on the cached features of studies ``tr``."""
    if cfg.reinit:
        _reset_head(model, cfg.seed)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(tr) // cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("backbone.")}
    rng = random.Random(cfg.seed)
    yt, wt = torch.from_numpy(np.nan_to_num(y)).float(), torch.from_numpy(w).float()
    model.train()
    model.backbone.eval()
    order = list(map(int, tr))
    for _ in range(cfg.epochs):
        rng.shuffle(order)
        for s in range(steps_per_epoch):
            idx = order[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            x, sl, po, mask = _batch(feats, idx, rng, True, cfg.feat_dropout)
            logits = model.forward_features(x.to(device), sl.to(device), po.to(device), mask.to(device))
            loss = composite_loss(logits.float(), yt[idx].to(device), wt[idx].to(device), cfg.auc_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in ema:
                        ema[n].mul_(cfg.ema_decay).add_(p.detach(), alpha=1 - cfg.ema_decay)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ema:
                p.copy_(ema[n])
    model.eval()
    return model


@torch.no_grad()
def predict_head(model: KneeMIL, feats: list[dict], idx: np.ndarray, device: torch.device,
                 batch_size: int = 64) -> np.ndarray:
    model.eval()
    rng = random.Random(0)
    out = []
    idx = list(map(int, idx))
    for s in range(0, len(idx), batch_size):
        x, sl, po, mask = _batch(feats, idx[s: s + batch_size], rng, False, 0.0)
        out.append(torch.sigmoid(model.forward_features(x.to(device), sl.to(device), po.to(device),
                                                        mask.to(device)).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 12), np.float32)
