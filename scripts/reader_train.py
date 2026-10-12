"""Train or evaluate one fold of the public 2.5-D ConvNeXt reader on a Kaggle T4.

    python scripts/reader_train.py --src <reader src dir> --cache <dir>[,<dir>...] \
        --labels labels_l1.csv --fold 0 --out runs/l1 [--eval-ckpt cnxt_v0_fold0.pt]

The model, dataset, windowing, augmentation, schedule, EMA and defaults are the reader's own
(goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache-2.0: ``knee.py`` and ``train.py``), so a run
here differs from the published reader only in its targets. Changes for a 16 GB T4, none of
which alter the optimisation target:

* fp16 autocast with a GradScaler instead of bf16 (T4 has no bf16 units).
* The batch of ``--bs`` studies is split into ``--micro`` sized pieces with gradient
  accumulation. The model has no batch-statistics layers, so the step equals the full batch.
* Volumes come from the compressed per-series ``.npz`` cache written by the reader's own
  ``preprocess.load_series`` (stage ``reader_cache``).
* The gold set is scored every epoch, the weak validation fold only after the last epoch.

``--eval-ckpt`` skips training and scores a published checkpoint the same way, which gives the
control arm for the label comparison. Outputs follow the reader: ``fold<k>.pt`` with ``{"model",
"args"}`` (loadable by the reader's ``infer.py``) and ``oof_fold<k>.csv`` with logits.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), aucs


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU only for smoke tests
AMP = dict(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda")


@torch.no_grad()
def predict(model, loader, res):
    model.eval()
    out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.to(DEV, non_blocking=True), mask.to(DEV), pos.to(DEV)
        with torch.autocast(**AMP):
            out.append(model(x, mask, pos, res=res).float().cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory with the reader's knee.py")
    ap.add_argument("--data", required=True, help="competition dir (train_series.csv)")
    ap.add_argument("--cache", required=True, help="reader_cache dir(s), comma-separated")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--eval-ckpt", default=None, help="score this checkpoint instead of training")
    # the reader's train.py defaults
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")  # as in the published args
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k_eval", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--seed", type=int, default=0)
    # T4 port
    ap.add_argument("--micro", type=int, default=2, help="studies per forward pass (gradient accumulation)")
    ap.add_argument("--eval-bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="timing probe: only this many training studies")
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests without downloading weights")
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    from knee import LABELS, KneeNet, StudyDataset, build_study_table  # noqa: PLC0415

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"log_fold{a.fold}.txt", "a")

    def say(msg):
        print(f"[reader f{a.fold}] {msg}", flush=True)
        log.write(msg + "\n")
        log.flush()

    say(json.dumps(vars(a)))
    paths, metas = {}, []
    for d in a.cache.split(","):
        d = Path(d)
        metas.append(pd.read_csv(d / "meta.csv"))
        paths.update({p.stem: str(p) for p in d.glob("*.npz")})
    meta = pd.concat(metas).drop_duplicates("series", keep="last")
    meta = meta[meta.series.isin(paths)]
    table = build_study_table(pd.read_csv(Path(a.data) / "train_series.csv"), dict(zip(meta.series, meta.n.fillna(0))))
    loader = lambda s: np.load(paths[s])["v"]  # noqa: E731
    say(f"cache: {len(paths)} series, {len(table)} studies")

    lab = pd.read_csv(a.labels, index_col=0)
    lab = lab[lab.index.isin(table)]
    gold, weak = lab[lab.is_gold == 1], lab[lab.is_gold == 0]
    tr, va = weak[weak.fold != a.fold], weak[weak.fold == a.fold]
    if a.limit:
        tr = tr.iloc[:a.limit]
    say(f"train {len(tr)}  val {len(va)}  gold {len(gold)}")

    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    eval_sets = {"gold": gold, "val": va}
    eval_dl = {k: torch.utils.data.DataLoader(StudyDataset(v.index, table, None, None, a.k_eval, loader=loader),
                                              batch_size=a.eval_bs, shuffle=False, **dl_kw)
               for k, v in eval_sets.items() if len(v)}
    gold_y = gold[[f"gold_{c}" for c in LABELS]].values

    def score_and_save(model, res, tag):
        preds = {k: predict(model, dl, res) for k, dl in eval_dl.items()}
        g_auc, g_each = macro_auc(gold_y, preds["gold"])
        msg = f"{tag} gold {g_auc:.4f}"
        if "val" in preds:
            v_auc, _ = macro_auc((va[LABELS].values > 0.5).astype(int), preds["val"])
            msg += f" | val {v_auc:.4f}"
        say(msg)
        say("gold per class: " + " ".join(f"{c[:6]}={x:.3f}" for c, x in zip(LABELS, g_each)))
        rows = []
        for k, v in eval_sets.items():
            if k in preds:
                df = pd.DataFrame(preds[k], index=v.index, columns=LABELS)
                df.insert(0, "set", k)
                rows.append(df)
        pd.concat(rows).to_csv(out / f"oof_fold{a.fold}.csv")
        (out / f"metrics_fold{a.fold}.json").write_text(json.dumps(
            {"gold_macro": g_auc, "gold_per_class": dict(zip(LABELS, g_each)), "n_gold": len(gold),
             "val_macro": v_auc if "val" in preds else None, "tag": tag}, indent=1))

    if a.eval_ckpt:
        ck = torch.load(a.eval_ckpt, map_location="cpu", weights_only=False)
        say("published args: " + json.dumps(ck.get("args", {})))
        model = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
        model.load_state_dict(ck["model"])
        model = model.to(DEV).to(memory_format=torch.channels_last)
        score_and_save(model, ck["args"]["res"], f"published {Path(a.eval_ckpt).name}")
        return

    accum = max(1, a.bs // a.micro)
    train_dl = torch.utils.data.DataLoader(
        StudyDataset(tr.index, table, None, tr[LABELS].values.astype(np.float32), a.k, train=True, loader=loader),
        batch_size=a.micro, shuffle=True, drop_last=True, **dl_kw)
    model = KneeNet(a.backbone, pretrained=not a.no_pretrained).to(DEV).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_params = list(model.enc.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=a.wd)
    per_epoch = len(train_dl) // accum
    steps, warm = a.epochs * per_epoch, per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler(DEV.type, enabled=DEV.type == "cuda")
    step = 0
    for ep in range(a.epochs):
        model.train()
        t0, tot, n, t_data = time.time(), 0.0, 0, 0.0
        it = iter(train_dl)
        for s in range(per_epoch):
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                td = time.time()
                x, mask, pos, y = next(it)
                t_data += time.time() - td
                x, mask, pos, y = x.to(DEV, non_blocking=True), mask.to(DEV), pos.to(DEV), y.to(DEV)
                with torch.autocast(**AMP):
                    logit = model(x, mask, pos, res=a.res, train=True)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y)
                scaler.scale(loss / accum).backward()
                tot += loss.item() / accum
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            d = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            n += 1
            if n % (10 if a.limit else 50) == 0:
                print(f"[reader f{a.fold}]   ep {ep} step {n}/{per_epoch} loss {tot / n:.4f} "
                      f"{(time.time() - t0) / n:.2f}s/step (data wait {t_data / n:.2f}s)", flush=True)
        g = predict(ema, eval_dl["gold"], a.res)
        say(f"ep {ep:2d} loss {tot / max(n, 1):.4f} time {time.time() - t0:.0f}s | gold {macro_auc(gold_y, g)[0]:.4f}")
    torch.save({"model": ema.state_dict(), "args": {**vars(a), "t4_port": True}}, out / f"fold{a.fold}.pt")
    score_and_save(ema, a.res, f"trained {a.epochs} ep")


if __name__ == "__main__":
    main()
