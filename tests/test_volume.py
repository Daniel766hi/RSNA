import numpy as np
import pytest

from conftest import AX, AX_F, COR, COR_F, SAG, SAG_T, make_study, write_series
from kneemri.volume import (
    VolumeConfig, build_study, canonicalize, load_series, load_study, mirror_to_left, physical_resample,
    resolve_laterality, sample_indices, save_study,
)

# expected sign of the intensity gradient along (slice, vertical, horizontal) after canonicalisation
EXPECTED = {"Sagittal": (+1, -1, +1), "Coronal": (+1, -1, +1), "Axial": (+1, +1, +1)}


def _grad_signs(px):
    s = np.sign(px[-1].mean() - px[0].mean())
    v = np.sign(px[:, -1, :].mean() - px[:, 0, :].mean())
    h = np.sign(px[:, :, -1].mean() - px[:, :, 0].mean())
    return int(s), int(v), int(h)


@pytest.mark.parametrize("plane,dirs", [("Sagittal", SAG), ("Sagittal", SAG_T), ("Coronal", COR),
                                        ("Coronal", COR_F), ("Axial", AX), ("Axial", AX_F)])
def test_canonical_orientation_is_acquisition_independent(tmp_path, plane, dirs):
    write_series(tmp_path / "s", *dirs, shuffle=True)
    vol, axes = load_series(tmp_path / "s")
    px, _ = canonicalize(vol.pixels, vol.spacing, axes, plane)
    assert _grad_signs(px) == EXPECTED[plane]


def test_slices_sorted_by_position_not_instance_number(tmp_path):
    write_series(tmp_path / "s", *SAG, n_slices=10, shuffle=True)
    vol, axes = load_series(tmp_path / "s")
    means = vol.pixels.mean(axis=(1, 2))
    assert np.all(np.diff(means) > 0) or np.all(np.diff(means) < 0)


def test_mirror_flips_patient_x_only():
    px = np.random.rand(4, 5, 6)
    assert np.array_equal(mirror_to_left(px, "Sagittal"), px[::-1])
    assert np.array_equal(mirror_to_left(px, "Coronal"), px[:, :, ::-1])


def test_laterality_tag_then_geometry():
    assert resolve_laterality(["R", None, "R"], [50.0], 15) == ("R", "tag")
    assert resolve_laterality([None], [40.0, 38.0], 15) == ("L", "geometry")   # +x = patient left
    assert resolve_laterality([None], [-40.0], 15) == ("R", "geometry")
    assert resolve_laterality([None], [3.0], 15) == (None, "unknown")


def test_physical_resample_scale():
    # 100 x 100 px at 2 mm = 200 mm FOV; a 100 mm crop should keep the central half
    stack = np.zeros((1, 100, 100), np.float32)
    stack[:, 25:75, 25:75] = 1.0
    out = physical_resample(stack, (2.0, 2.0), 100.0, 50)
    assert out.shape == (1, 50, 50)
    assert out[0, 5:45, 5:45].min() > 0.9            # the 100 mm square fills the crop
    wide = physical_resample(stack, (2.0, 2.0), 400.0, 40)   # crop larger than the image pads
    assert wide.shape == (1, 40, 40) and wide[0, 0, 0] == 0.0


def test_sample_indices_span():
    idx = sample_indices(30, 5, (0.0, 1.0))
    assert idx[0] == 0 and idx[-1] == 29 and len(idx) == 5


def test_build_study_roundtrip_and_mirroring(tmp_path):
    cfg = VolumeConfig(img=32, crop_mm=60)
    left = make_study(tmp_path, "L1", laterality="L")
    right = make_study(tmp_path, "R1", laterality="R")
    a_l, m_l = build_study(tmp_path / "L1", left, cfg)
    a_r, m_r = build_study(tmp_path / "R1", right, cfg)
    assert m_l.side == "L" and not m_l.mirrored and m_r.side == "R" and m_r.mirrored
    for slot in ("sag_fs", "sag_anat", "cor_fs", "cor_anat", "ax_fs"):
        assert a_l[f"{slot}_img"].dtype == np.uint8 and a_l[f"{slot}_img"].shape[1:] == (32, 32)
    # x-gradient: along slices on sagittal, along the horizontal axis on coronal; mirrored for R
    sl, sr = a_l["sag_fs_img"].astype(float), a_r["sag_fs_img"].astype(float)
    assert np.sign(sl[-1].mean() - sl[0].mean()) == -np.sign(sr[-1].mean() - sr[0].mean())
    cl, cr = a_l["cor_fs_img"].astype(float), a_r["cor_fs_img"].astype(float)
    assert np.sign(cl[..., -1].mean() - cl[..., 0].mean()) == -np.sign(cr[..., -1].mean() - cr[..., 0].mean())

    save_study(tmp_path / "c" / "L1.npz", a_l, m_l)
    arrays, meta = load_study(tmp_path / "c" / "L1.npz")
    assert meta["side"] == "L" and set(arrays) == set(a_l)
    assert np.array_equal(arrays["sag_fs_img"], a_l["sag_fs_img"])
