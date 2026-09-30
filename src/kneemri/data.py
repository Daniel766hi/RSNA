"""Cached study -> 2.5-D windows -> padded batches.

A *window* is three neighbouring sampled slices of one slot, stacked as RGB channels (the 2.5-D
input used by the RSNA 2022-2024 winners and by the public CoAtNets). Every window carries its
slot id and its normalised slice position, which the model embeds (lever L2).

Augmentations are small affine and intensity changes applied consistently to a whole study.
There are no flips: after laterality normalisation, a horizontal flip would swap medial and
lateral, and a vertical one would swap femur and tibia. Window and slot dropout make the
attention pool robust to missing or partial series, which are common across the 5+ sites.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .series import DEFAULT_SLOTS, SlotSpec
from .volume import load_study


def study_to_windows(
    arrays: dict[str, np.ndarray], slots: tuple[SlotSpec, ...] = DEFAULT_SLOTS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (windows uint8 [W,3,H,W], slot ids int64 [W], positions float32 [W])."""
    wins, sids, poss = [], [], []
    for k, spec in enumerate(slots):
        img = arrays.get(f"{spec.name}_img")
        if img is None or len(img) == 0:
            continue
        pos = arrays.get(f"{spec.name}_pos", np.linspace(0, 1, len(img), dtype=np.float32))
        n = len(img)
        prev_idx = np.clip(np.arange(n) - 1, 0, n - 1)
        next_idx = np.clip(np.arange(n) + 1, 0, n - 1)
        wins.append(np.stack([img[prev_idx], img, img[next_idx]], axis=1))
        sids.append(np.full(n, k, dtype=np.int64))
        poss.append(pos.astype(np.float32))
    if not wins:
        return np.zeros((0, 3, 1, 1), np.uint8), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(wins), np.concatenate(sids), np.concatenate(poss)


def augment(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Study-consistent affine + intensity augmentation of float windows in [0,1], shape [W,3,H,W]."""
    ang = math.radians(rng.uniform(-12, 12))
    sc = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)
    cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=x.dtype).expand(x.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = math.exp(rng.uniform(-0.25, 0.25))
    x = x.clamp(0, 1) ** gamma
    x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.3:
        x = x + torch.randn_like(x) * rng.uniform(0.0, 0.03)
    return x


class KneeStudyDataset(Dataset):
    """Studies from the ``.npz`` cache with soft targets ``y`` and per-cell weights ``w``."""

    def __init__(
        self,
        paths: list[str | Path],
        y: np.ndarray | None = None,
        w: np.ndarray | None = None,
        slots: tuple[SlotSpec, ...] = DEFAULT_SLOTS,
        train: bool = False,
        img_size: int | None = None,
        window_dropout: float = 0.2,
        slot_dropout: float = 0.1,
        max_windows: int | None = None,
        seed: int = 0,
    ):
        self.paths = [Path(p) for p in paths]
        n, k = len(self.paths), 12
        self.y = np.zeros((n, k), np.float32) if y is None else np.nan_to_num(y.astype(np.float32))
        self.w = np.ones((n, k), np.float32) if w is None else w.astype(np.float32)
        if y is not None:
            self.w = np.where(np.isnan(y), 0.0, self.w).astype(np.float32)
        self.slots, self.train, self.img_size = slots, train, img_size
        self.window_dropout, self.slot_dropout, self.max_windows = window_dropout, slot_dropout, max_windows
        self.seed = seed
        self.rng = random.Random(seed)
        self._calls = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        arrays, _ = load_study(self.paths[i])
        return self.from_arrays(arrays, i)

    def from_arrays(self, arrays: dict[str, np.ndarray], i: int = 0) -> dict:
        if self.train:
            # DataLoader workers get copies of this object; torch's per-worker, per-epoch seed
            # keeps their augmentation streams distinct
            self._calls += 1
            self.rng.seed((torch.initial_seed() + self.seed + 1_000_003 * i + self._calls) % (2**63))
        win, sid, pos = study_to_windows(arrays, self.slots)
        keep = np.ones(len(win), bool)
        if self.train and len(win) > 0:
            present = np.unique(sid)
            if len(present) > 1:
                for s in present:
                    if self.rng.random() < self.slot_dropout:
                        keep[sid == s] = False
                if not keep.any():
                    keep[sid == present[0]] = True
            drop = np.array([self.rng.random() < self.window_dropout for _ in range(len(win))])
            if (keep & ~drop).any():
                keep &= ~drop
        if self.max_windows is not None and keep.sum() > self.max_windows:
            # train: a fresh random subset every epoch (the position embedding tells the model
            # where each window sits); eval: evenly spaced, deterministic
            idx = np.flatnonzero(keep)
            if self.train:
                sel = np.sort(self.rng.sample(range(len(idx)), self.max_windows))
            else:
                sel = np.round(np.linspace(0, len(idx) - 1, self.max_windows)).astype(int)
            keep[:] = False
            keep[idx[sel]] = True
        win, sid, pos = win[keep], sid[keep], pos[keep]
        x = torch.from_numpy(win).float().div_(255.0)
        if self.img_size is not None and x.shape[0] > 0 and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False, antialias=True)
        if self.train and x.shape[0] > 0:
            x = augment(x, self.rng)
        x = (x - 0.5) / 0.25
        return {
            "x": x,
            "slot": torch.from_numpy(sid),
            "pos": torch.from_numpy(pos),
            "y": torch.from_numpy(self.y[i]),
            "w": torch.from_numpy(self.w[i]),
            "index": i,
        }


def collate(batch: list[dict]) -> dict:
    """Pad windows to the longest study in the batch; ``mask`` is True on real windows."""
    b = len(batch)
    w_max = max(1, max(item["x"].shape[0] for item in batch))
    shape = next((item["x"].shape[1:] for item in batch if item["x"].shape[0] > 0), None)
    if shape is None:
        raise ValueError("batch contains no windows at all")
    x = torch.zeros((b, w_max, *shape))
    slot = torch.zeros((b, w_max), dtype=torch.long)
    pos = torch.zeros((b, w_max))
    mask = torch.zeros((b, w_max), dtype=torch.bool)
    for j, item in enumerate(batch):
        n = item["x"].shape[0]
        if n == 0:
            continue
        x[j, :n], slot[j, :n], pos[j, :n], mask[j, :n] = item["x"], item["slot"], item["pos"], True
    return {
        "x": x, "slot": slot, "pos": pos, "mask": mask,
        "y": torch.stack([item["y"] for item in batch]),
        "w": torch.stack([item["w"] for item in batch]),
        "index": torch.tensor([item["index"] for item in batch]),
    }
