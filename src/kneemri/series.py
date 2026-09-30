"""Pick one series per *slot* (plane x contrast) for every study.

A study holds 3-8 series with different planes and contrasts, so the model sees a fixed set of
slots instead of the raw series list. Each finding is best seen on particular slots: ACL and the
menisci on sagittal, MCL on coronal, Hoffa/effusion-synovitis on sagittal fluid-sensitive (see
docs/research_review.md §3.1 and §3.7). The per-finding attention head in ``model.py`` learns
which slots to read, using the slot embedding.

Slot choice uses only the host-provided descriptors in ``train_series.csv`` / ``test_series.csv``
(``Anatomical_Plane``, ``Fluid_Sensitive``, ``Fat_Suppression``), so it is identical at train and
test time and needs no DICOM header pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .constants import SERIES_COL


@dataclass(frozen=True)
class SlotSpec:
    name: str
    plane: str                  # "Sagittal" | "Coronal" | "Axial"
    fluid: int | None           # preferred Fluid_Sensitive (None = no preference)
    fatsat: int | None          # preferred Fat_Suppression (None = no preference)
    n_slices: int               # slices sampled from the chosen series


# Sagittal carries the cruciates, menisci, Hoffa's fat pad and the suprapatellar recess, so it
# gets two slots and the most slices. The "anat" slots prefer non-fat-suppressed sequences
# (PD / T1), which show cartilage and meniscal morphology; the "fs" slots prefer fluid-sensitive
# fat-suppressed sequences, which show oedema, contusion, effusion and tears.
DEFAULT_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("sag_fs", "Sagittal", 1, 1, 16),
    SlotSpec("sag_anat", "Sagittal", None, 0, 12),
    SlotSpec("cor_fs", "Coronal", 1, 1, 12),
    SlotSpec("cor_anat", "Coronal", None, 0, 8),
    SlotSpec("ax_fs", "Axial", 1, None, 10),
)


def slot_names(slots: tuple[SlotSpec, ...] = DEFAULT_SLOTS) -> list[str]:
    return [s.name for s in slots]


def _score(row: pd.Series, spec: SlotSpec) -> float:
    score = 0.0
    if spec.fluid is not None and int(row.get("Fluid_Sensitive", -1)) == spec.fluid:
        score += 2.0
    if spec.fatsat is not None and int(row.get("Fat_Suppression", -1)) == spec.fatsat:
        score += 2.0
    # Prefer series with enough slices to cover the joint; saturate so a 300-slice 3-D
    # acquisition does not beat a clinical 2-D series on slice count alone.
    n = row.get("n_slices")
    if n is not None and not pd.isna(n):
        score += min(float(n), 40.0) / 100.0
    return score


def select_slots(
    study_series: pd.DataFrame, slots: tuple[SlotSpec, ...] = DEFAULT_SLOTS
) -> dict[str, str | None]:
    """Greedy slot assignment for one study.

    ``study_series`` has one row per series with ``SeriesInstanceUID``, ``Anatomical_Plane``,
    ``Fluid_Sensitive``, ``Fat_Suppression`` and optionally ``n_slices``. A series fills at most
    one slot unless its plane has no other series; a plane with no series leaves its slots
    empty (the model masks missing slots).
    """
    chosen: dict[str, str | None] = {}
    used: set[str] = set()
    for spec in slots:
        cand = study_series[study_series["Anatomical_Plane"] == spec.plane]
        if cand.empty:
            chosen[spec.name] = None
            continue
        scores = cand.apply(lambda r: _score(r, spec) - (10.0 if r[SERIES_COL] in used else 0.0), axis=1)
        best = cand.loc[scores.idxmax(), SERIES_COL]
        chosen[spec.name] = str(best)
        used.add(str(best))
    return chosen
