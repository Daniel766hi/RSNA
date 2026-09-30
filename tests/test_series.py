import pandas as pd

from kneemri.series import DEFAULT_SLOTS, select_slots


def _row(sid, plane, fluid, fs, n=30):
    return {"SeriesInstanceUID": sid, "Anatomical_Plane": plane, "Fluid_Sensitive": fluid, "Fat_Suppression": fs, "n_slices": n}


def test_slots_prefer_matching_contrast_and_do_not_reuse():
    df = pd.DataFrame([
        _row("sag_pd", "Sagittal", 1, 0), _row("sag_pdfs", "Sagittal", 1, 1),
        _row("cor_pdfs", "Coronal", 1, 1), _row("cor_t1", "Coronal", 0, 0), _row("ax_pdfs", "Axial", 1, 1),
    ])
    got = select_slots(df, DEFAULT_SLOTS)
    assert got == {"sag_fs": "sag_pdfs", "sag_anat": "sag_pd", "cor_fs": "cor_pdfs", "cor_anat": "cor_t1", "ax_fs": "ax_pdfs"}


def test_single_series_plane_is_reused_and_missing_plane_is_none():
    df = pd.DataFrame([_row("sag", "Sagittal", 1, 1), _row("cor", "Coronal", 1, 1)])
    got = select_slots(df, DEFAULT_SLOTS)
    assert got["sag_fs"] == got["sag_anat"] == "sag"
    assert got["ax_fs"] is None
