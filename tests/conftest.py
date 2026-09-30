"""Synthetic DICOM studies for tests.

Pixel intensity is a known linear function of the patient (LPS) position,
``I = 1000 + GX*x + GY*y + GZ*z``, so after canonicalisation the sign of the intensity gradient
along each array axis tells us which patient direction that axis points to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GX, GY, GZ = 1.0, 2.0, 3.0


def world_intensity(pts: np.ndarray) -> np.ndarray:
    return 1000.0 + GX * pts[..., 0] + GY * pts[..., 1] + GZ * pts[..., 2]


def write_series(
    out_dir: Path, row_dir, col_dir, n_slices=8, rows=40, cols=48, spacing=(2.0, 2.0), gap=3.0,
    origin=(0.0, 0.0, 0.0), laterality=None, shuffle=True, seed=0,
) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    out_dir.mkdir(parents=True, exist_ok=True)
    r, c = np.asarray(row_dir, float), np.asarray(col_dir, float)
    n = np.cross(r, c)
    origin = np.asarray(origin, float)
    # centre the volume on ``origin``
    start = origin - r * spacing[1] * (cols - 1) / 2 - c * spacing[0] * (rows - 1) / 2 - n * gap * (n_slices - 1) / 2
    order = np.random.default_rng(seed).permutation(n_slices) if shuffle else np.arange(n_slices)
    series_uid = generate_uid()
    for k, s in enumerate(order):
        ipp = start + n * gap * s
        ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        pts = ipp + ii[..., None] * c * spacing[0] + jj[..., None] * r * spacing[1]
        img = np.clip(world_intensity(pts), 0, 65535).astype(np.uint16)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        ds.InstanceNumber = int(k + 1)            # deliberately NOT the spatial order
        ds.ImageOrientationPatient = [float(v) for v in (*r, *c)]
        ds.ImagePositionPatient = [float(v) for v in ipp]
        ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
        if laterality:
            ds.Laterality = laterality
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = img.tobytes()
        path = out_dir / f"{ds.SOPInstanceUID}.dcm"
        try:
            ds.save_as(str(path), enforce_file_format=True)
        except TypeError:  # pydicom < 3
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(str(path), write_like_original=False)


# plane -> (row_dir, col_dir) variants, including flipped / transposed acquisitions
SAG = ((0, -1, 0), (0, 0, -1))      # row along -y (anterior), col inferior
SAG_T = ((0, 0, 1), (0, 1, 0))      # transposed and flipped
COR = ((1, 0, 0), (0, 0, -1))
COR_F = ((-1, 0, 0), (0, 0, 1))
AX = ((1, 0, 0), (0, 1, 0))
AX_F = ((-1, 0, 0), (0, -1, 0))


def make_study(root: Path, study_id: str, laterality: str | None = "L", x_offset: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Five series (sag FS, sag PD, cor FS, cor T1, ax FS); returns the series table rows."""
    specs = [
        ("s1", "Sagittal", 1, 1, SAG), ("s2", "Sagittal", 1, 0, SAG_T),
        ("s3", "Coronal", 1, 1, COR), ("s4", "Coronal", 0, 0, COR_F), ("s5", "Axial", 1, 1, AX_F),
    ]
    rows = []
    for k, (name, plane, fluid, fs, (r, c)) in enumerate(specs):
        sid = f"{study_id}.{name}"
        write_series(root / study_id / sid, r, c, origin=(x_offset, 0, 0), laterality=laterality, seed=seed + k)
        rows.append({"StudyInstanceUID": study_id, "SeriesInstanceUID": sid, "Fluid_Sensitive": fluid,
                     "Fat_Suppression": fs, "Anatomical_Plane": plane, "n_slices": 8})
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_study(tmp_path):
    series = make_study(tmp_path, "1.2.3", laterality="L")
    return tmp_path, series
