"""DICOM series -> canonical, physically scaled, laterality-normalised slice stacks.

The pipeline for one series:

1. Read every slice. Sort by position along the slice normal (``ImagePositionPatient`` .
   normal), falling back to ``InstanceNumber``.
2. **Standardise orientation** from ``ImageOrientationPatient``. Axes are transposed and
   flipped so that every plane has the same display convention (LPS patient coordinates):

   ========  ================  ==============  ================
   plane     slice axis        vertical axis   horizontal axis
   ========  ================  ==============  ================
   Sagittal  +x (to L)         -z (inferior)   +y (posterior)
   Coronal   +y (posterior)    -z (inferior)   +x (to L)
   Axial     +z (superior)     +y (posterior)  +x (to L)
   ========  ================  ==============  ================

3. **Mirror right knees** so that every study looks like a left knee: flip the +x axis. On a
   left knee, +x points lateral, so after mirroring the patient x-axis always runs
   medial -> lateral. This is normalisation, not augmentation: flips are never used as
   augmentation because they would swap medial and lateral labels.
4. **Resample at a fixed physical scale.** Take a ``crop_mm`` square around the (optionally
   re-centred) joint and resize it to ``img`` px, so anatomy covers the same number of pixels
   at every site and resolution.
5. **Sample** ``n`` slices evenly over ``span`` of the series, and record each slice's
   normalised position (0..1) along the canonical slice axis. For sagittal slots that
   position runs medial -> lateral: the signal the permutation-invariant public models cannot
   see (docs/research_review.md §2, lever L2).
6. **Normalise intensity** robustly per slot (percentiles) to uint8 for compact caching.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SERIES_COL
from .series import DEFAULT_SLOTS, SlotSpec, select_slots

# canonical (slice, vertical, horizontal) target directions per plane, LPS coordinates
_CANON = {
    "Sagittal": (np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0])),
    "Coronal": (np.array([0, 1.0, 0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0])),
    "Axial": (np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])),
}


@dataclass
class VolumeConfig:
    img: int = 384                      # output pixel grid
    crop_mm: float = 140.0              # physical field of view of the crop (L4: try 120)
    span: tuple[float, float] = (0.04, 0.96)  # fraction of the series the samples cover
    recenter: bool = False              # shift the crop centre towards the foreground centroid
    max_shift_mm: float = 25.0
    mirror_right: bool = True           # mirror right knees to the left-knee convention
    pct: tuple[float, float] = (1.0, 99.5)
    geom_min_offset_mm: float = 15.0    # |x| of the volume centre needed to trust the geometry
    slots: tuple[SlotSpec, ...] = field(default_factory=lambda: DEFAULT_SLOTS)


@dataclass
class SeriesVolume:
    pixels: np.ndarray            # float32 [S, H, W], canonical orientation
    spacing: tuple[float, float]  # (vertical, horizontal) mm per pixel
    centre_lps: np.ndarray        # physical centre of the volume, LPS mm
    laterality_tag: str | None    # "L" / "R" from DICOM tags, if present


# --------------------------------------------------------------------------------------------
# reading


def _read_slices(series_dir: Path):
    import pydicom

    out = []
    for f in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), force=True)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:  # corrupt slice or missing codec: skip, the rest may be usable
            continue
        if arr.ndim == 3:  # multi-frame / RGB: keep the first channel or frame
            arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        out.append((ds, arr * slope + inter))
    return out


def _tag_side(datasets) -> str | None:
    votes = []
    for ds in datasets:
        for tag in ("Laterality", "ImageLaterality"):
            v = str(getattr(ds, tag, "") or "").strip().upper()
            if v in ("L", "R"):
                votes.append(v)
    if not votes:
        return None
    return max(set(votes), key=votes.count)


def load_series(series_dir: str | Path) -> tuple[SeriesVolume, tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Read one series into a sorted float32 stack (not yet canonical) plus its (row, col, normal) axes."""
    items = _read_slices(Path(series_dir))
    if not items:
        return None
    # keep the modal in-plane shape (localisers and odd frames sometimes sneak in)
    shapes = [a.shape for _, a in items]
    mode = max(set(shapes), key=shapes.count)
    items = [(d, a) for d, a in items if a.shape == mode]

    ds0 = items[0][0]
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        row_dir = np.asarray(iop[:3], dtype=np.float64)   # direction of increasing column index
        col_dir = np.asarray(iop[3:], dtype=np.float64)   # direction of increasing row index
    else:
        row_dir, col_dir = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    normal = np.cross(row_dir, col_dir)

    def key(item):
        d = item[0]
        ipp = getattr(d, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))
        return float(getattr(d, "InstanceNumber", 0) or 0)

    items.sort(key=key)
    pixels = np.stack([a for _, a in items]).astype(np.float32)
    ps = getattr(ds0, "PixelSpacing", None)
    sp = (float(ps[0]), float(ps[1])) if ps is not None and len(ps) == 2 else (0.5, 0.5)

    ipps = [getattr(d, "ImagePositionPatient", None) for d, _ in items]
    if all(p is not None and len(p) == 3 for p in ipps):
        first, last = np.asarray(ipps[0], float), np.asarray(ipps[-1], float)
        h, w = mode
        in_plane = row_dir * sp[1] * (w - 1) / 2 + col_dir * sp[0] * (h - 1) / 2
        centre = (first + last) / 2 + in_plane
    else:
        centre = np.zeros(3)

    return SeriesVolume(
        pixels=pixels, spacing=sp, centre_lps=centre, laterality_tag=_tag_side([d for d, _ in items]),
    ), (row_dir, col_dir, normal)


# --------------------------------------------------------------------------------------------
# geometry


def canonicalize(
    pixels: np.ndarray, spacing: tuple[float, float], axes: tuple[np.ndarray, np.ndarray, np.ndarray], plane: str
) -> tuple[np.ndarray, tuple[float, float]]:
    """Transpose/flip a sorted stack ``[S, rows, cols]`` into the canonical orientation of ``plane``.

    ``axes`` = (row_dir, col_dir, normal) from ``ImageOrientationPatient``: moving along the
    column index moves along ``row_dir``, moving along the row index moves along ``col_dir``,
    and the stack is sorted ascending along ``normal``.
    """
    row_dir, col_dir, normal = axes
    t_slice, t_vert, t_horiz = _CANON.get(plane, _CANON["Sagittal"])
    vert_vec, horiz_vec = col_dir, row_dir
    sp_v, sp_h = spacing
    # transpose in-plane if the array's vertical axis is closer to the canonical horizontal
    if abs(np.dot(vert_vec, t_horiz)) > abs(np.dot(vert_vec, t_vert)):
        pixels = pixels.transpose(0, 2, 1)
        vert_vec, horiz_vec = horiz_vec, vert_vec
        sp_v, sp_h = sp_h, sp_v
    if np.dot(vert_vec, t_vert) < 0:
        pixels = pixels[:, ::-1, :]
    if np.dot(horiz_vec, t_horiz) < 0:
        pixels = pixels[:, :, ::-1]
    if np.dot(normal, t_slice) < 0:
        pixels = pixels[::-1]
    return np.ascontiguousarray(pixels), (sp_v, sp_h)


def mirror_to_left(pixels: np.ndarray, plane: str) -> np.ndarray:
    """Flip the patient x-axis of a canonical stack so that a right knee looks like a left knee."""
    if plane == "Sagittal":
        return np.ascontiguousarray(pixels[::-1])          # slice axis is +x
    return np.ascontiguousarray(pixels[:, :, ::-1])        # horizontal axis is +x


def resolve_laterality(tags: list[str | None], centres_x: list[float], min_offset_mm: float) -> tuple[str | None, str]:
    """Study-level knee side: majority DICOM tag, else the sign of the volume centre's x.

    In LPS coordinates +x is the patient's left, so a knee imaged off-isocentre towards +x is a
    left knee. Returns (side, source) with source in {"tag", "geometry", "unknown"}.
    """
    t = [s for s in tags if s in ("L", "R")]
    if t:
        return max(set(t), key=t.count), "tag"
    xs = [x for x in centres_x if abs(x) >= min_offset_mm]
    if xs:
        return ("L" if np.median(xs) > 0 else "R"), "geometry"
    return None, "unknown"


def _foreground_shift(stack: np.ndarray, spacing: tuple[float, float], max_shift_mm: float) -> tuple[float, float]:
    """In-plane (vertical, horizontal) shift in mm from the frame centre to the foreground centroid."""
    s = stack.shape[0]
    mid = stack[max(0, s // 2 - 2): s // 2 + 3].mean(0)
    thr = mid.mean()
    mask = mid > thr
    if mask.sum() < 16:
        return 0.0, 0.0
    ys, xs = np.nonzero(mask)
    h, w = mid.shape
    dv = (ys.mean() - (h - 1) / 2) * spacing[0]
    dh = (xs.mean() - (w - 1) / 2) * spacing[1]
    return float(np.clip(dv, -max_shift_mm, max_shift_mm)), float(np.clip(dh, -max_shift_mm, max_shift_mm))


def physical_resample(
    stack: np.ndarray, spacing: tuple[float, float], crop_mm: float, img: int, shift_mm: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Crop ``crop_mm`` x ``crop_mm`` around the (shifted) centre of each slice and resize to ``img``."""
    import torch
    import torch.nn.functional as F

    s, h, w = stack.shape
    cy = (h - 1) / 2 + shift_mm[0] / spacing[0]
    cx = (w - 1) / 2 + shift_mm[1] / spacing[1]
    hv = crop_mm / 2 / spacing[0]
    hh = crop_mm / 2 / spacing[1]
    y0, y1 = int(round(cy - hv)), int(round(cy + hv))
    x0, x1 = int(round(cx - hh)), int(round(cx + hh))
    pad = (max(0, -x0), max(0, x1 - w), max(0, -y0), max(0, y1 - h))
    t = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32))[:, None]
    if any(pad):
        t = F.pad(t, pad, value=0.0)
        y0, y1, x0, x1 = y0 + pad[2], y1 + pad[2], x0 + pad[0], x1 + pad[0]
    t = t[:, :, y0:y1, x0:x1]
    t = F.interpolate(t, size=(img, img), mode="bilinear", align_corners=False, antialias=True)
    return t[:, 0].numpy()


def sample_indices(n_avail: int, n: int, span: tuple[float, float]) -> np.ndarray:
    lo, hi = span[0] * (n_avail - 1), span[1] * (n_avail - 1)
    return np.clip(np.round(np.linspace(lo, hi, n)), 0, n_avail - 1).astype(int)


def to_uint8(stack: np.ndarray, pct: tuple[float, float]) -> np.ndarray:
    lo, hi = np.percentile(stack, pct)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((stack - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


# --------------------------------------------------------------------------------------------
# study


@dataclass
class StudyMeta:
    study_id: str
    side: str | None
    side_source: str
    mirrored: bool
    slots: dict[str, str | None]


def build_study(
    study_dir: str | Path, study_series: pd.DataFrame, cfg: VolumeConfig | None = None
) -> tuple[dict[str, np.ndarray], StudyMeta]:
    """Decode one study into ``{"<slot>_img": uint8[n,H,W], "<slot>_pos": float32[n]}`` + meta.

    Missing or unreadable slots are simply absent from the dict.
    """
    cfg = cfg or VolumeConfig()
    study_dir = Path(study_dir)
    chosen = select_slots(study_series, cfg.slots)
    loaded: dict[str, tuple[SeriesVolume, tuple]] = {}
    for spec in cfg.slots:
        sid = chosen.get(spec.name)
        if sid is None:
            continue
        if sid in loaded:
            continue
        res = load_series(study_dir / sid)
        if res is not None:
            loaded[sid] = res

    side, source = resolve_laterality(
        [v.laterality_tag for v, _ in loaded.values()],
        [float(v.centre_lps[0]) for v, _ in loaded.values()],
        cfg.geom_min_offset_mm,
    )
    mirror = cfg.mirror_right and side == "R"

    out: dict[str, np.ndarray] = {}
    for spec in cfg.slots:
        sid = chosen.get(spec.name)
        if sid is None or sid not in loaded:
            continue
        vol, axes = loaded[sid]
        px, sp = canonicalize(vol.pixels, vol.spacing, axes, spec.plane)
        if mirror:
            px = mirror_to_left(px, spec.plane)
        idx = sample_indices(px.shape[0], spec.n_slices, cfg.span)
        shift = _foreground_shift(px, sp, cfg.max_shift_mm) if cfg.recenter else (0.0, 0.0)
        res = physical_resample(px[idx], sp, cfg.crop_mm, cfg.img, shift)
        out[f"{spec.name}_img"] = to_uint8(res, cfg.pct)
        denom = max(px.shape[0] - 1, 1)
        out[f"{spec.name}_pos"] = (idx / denom).astype(np.float32)

    meta = StudyMeta(str(study_dir.name), side, source, bool(mirror), chosen)
    return out, meta


def save_study(path: str | Path, arrays: dict[str, np.ndarray], meta: StudyMeta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.frombuffer(json.dumps(asdict(meta)).encode(), dtype=np.uint8), **arrays)


def load_study(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z.files else {}
    return arrays, meta


def count_slices(series_root: str | Path, series_df: pd.DataFrame, study_col: str = "StudyInstanceUID") -> pd.Series:
    """Number of .dcm files per series (cheap directory listing, no decode)."""
    root = Path(series_root)
    return series_df.apply(
        lambda r: sum(1 for _ in (root / str(r[study_col]) / str(r[SERIES_COL])).glob("*.dcm")), axis=1
    )
