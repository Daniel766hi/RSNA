"""Locate inputs inside a Kaggle notebook without hard-coding mount paths.

Kaggle has changed its ``/kaggle/input`` layout over time (``/kaggle/input/<slug>`` vs
``/kaggle/input/competitions/<slug>`` vs ``/kaggle/input/datasets/<user>/<slug>``), so every
stage searches instead of assuming a path.
"""

from __future__ import annotations

import os
from pathlib import Path

# overridable so the stage scripts can be simulated outside Kaggle
KAGGLE_ROOT = Path(os.environ.get("KNEEMRI_KAGGLE_ROOT", "/kaggle"))
INPUT = KAGGLE_ROOT / "input"
WORK = KAGGLE_ROOT / "working"


def find_competition_dir(root: Path = INPUT) -> Path:
    for p in sorted(root.rglob("train_series.csv")):
        return p.parent
    for p in sorted(root.rglob("test_series.csv")):
        return p.parent
    raise FileNotFoundError(f"competition data not found under {root}")


def find_files(pattern: str, root: Path = INPUT) -> list[Path]:
    return sorted(root.rglob(pattern))


def find_npz_cache(root: Path = INPUT, min_files: int = 100) -> Path:
    """The directory holding the most ``.npz`` study caches."""
    best, n_best = None, 0
    for d in {p.parent for p in root.rglob("*.npz")}:
        n = sum(1 for _ in d.glob("*.npz"))
        if n > n_best:
            best, n_best = d, n
    if best is None or n_best < min_files:
        raise FileNotFoundError(f"no .npz cache with >= {min_files} studies under {root}")
    return best


def find_npz_caches(names: list[str], root: Path = INPUT, min_files: int = 100) -> list[Path]:
    """Every ``.npz`` cache directory whose path contains one of ``names`` (cache kernel slugs)."""
    dirs = sorted({p.parent for p in root.rglob("*.npz") if any(n in str(p) for n in names)})
    dirs = [d for d in dirs if sum(1 for _ in d.glob("*.npz")) >= min_files]
    if not dirs:
        raise FileNotFoundError(f"no .npz cache for {names} under {root}")
    return dirs
