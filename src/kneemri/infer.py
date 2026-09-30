"""Offline, time-budgeted inference straight from DICOM.

Each study is decoded **once** (CPU workers, overlapped with GPU compute) and shared by every
checkpoint in the group. Decoding costs about as much as the model (0.6-1.8 s/study), so this
matters for the efficiency track and for fitting extra members into the 9 h budget.

Checkpoints of one run (folds) are averaged in logit space. Different runs or groups are fused
by rank mean in ``ensemble.py``. A study that fails to decode, or is not reached before the time
budget, gets 0.5 for every target: ties, never a crash.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import ID_COL, TARGETS
from .data import KneeStudyDataset, collate
from .model import KneeMIL, ModelConfig
from .volume import VolumeConfig, build_study


class DicomStudyDataset(Dataset):
    def __init__(self, study_ids: list[str], series_df: pd.DataFrame, series_root: str | Path,
                 vol_cfg: VolumeConfig, n_slots: int, img_size: int | None = None):
        self.ids = list(study_ids)
        self.groups = {k: g for k, g in series_df.groupby(ID_COL)}
        self.root = Path(series_root)
        self.vol_cfg = vol_cfg
        self.view = KneeStudyDataset([""] * len(self.ids), slots=vol_cfg.slots[:n_slots], img_size=img_size)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        sid = self.ids[i]
        try:
            arrays, _ = build_study(self.root / sid, self.groups[sid], self.vol_cfg)
            item = self.view.from_arrays(arrays, i)
        except Exception as exc:  # noqa: BLE001 - one bad study must not kill the submission
            print(f"[infer] decode failed for {sid}: {exc!r}", flush=True)
            item = None
        if item is None or item["x"].shape[0] == 0:
            item = {"x": torch.zeros(0, 3, 1, 1), "slot": torch.zeros(0, dtype=torch.long),
                    "pos": torch.zeros(0), "y": torch.zeros(len(TARGETS)), "w": torch.zeros(len(TARGETS)), "index": i}
        return item


def _safe_collate(batch: list[dict]) -> dict | None:
    ok = [b for b in batch if b["x"].shape[0] > 0]
    failed = [b["index"] for b in batch if b["x"].shape[0] == 0]
    return {"batch": collate(ok) if ok else None, "failed": failed}


def load_models(ckpts: list[str | Path], device: torch.device) -> list[KneeMIL]:
    models = []
    for p in ckpts:
        state = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**{**state["model_cfg"], "pretrained": False})
        m = KneeMIL(cfg)
        m.load_state_dict(state["state_dict"])
        models.append(m.to(device).eval())
    return models


@torch.no_grad()
def predict_dicom(
    study_ids: list[str],
    series_df: pd.DataFrame,
    series_root: str | Path,
    ckpts: list[str | Path],
    vol_cfg: VolumeConfig | None = None,
    img_size: int | None = None,
    batch_size: int = 2,
    num_workers: int = 4,
    time_budget_s: float | None = None,
) -> pd.DataFrame:
    """Predict probabilities for ``study_ids`` with the fold checkpoints ``ckpts`` (one group)."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(ckpts, device)
    vol_cfg = vol_cfg or VolumeConfig()
    ds = DicomStudyDataset(study_ids, series_df, series_root, vol_cfg, models[0].cfg.n_slots, img_size)
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers, collate_fn=_safe_collate)
    preds = np.full((len(study_ids), len(TARGETS)), 0.5, np.float32)
    for pack in dl:
        if time_budget_s is not None and time.time() - t0 > time_budget_s:
            print("[infer] time budget reached; remaining studies get 0.5", flush=True)
            break
        b = pack["batch"]
        if b is None:
            continue
        x, slot, pos, mask = (b[k].to(device) for k in ("x", "slot", "pos", "mask"))
        logits = []
        for m in models:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits.append(m(x, slot, pos, mask).float())
        preds[b["index"].numpy()] = torch.sigmoid(torch.stack(logits).mean(0)).cpu().numpy()
    out = pd.DataFrame(preds, columns=TARGETS)
    out.insert(0, ID_COL, list(study_ids))
    return out
