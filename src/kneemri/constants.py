"""Competition-wide constants."""

from __future__ import annotations

TARGETS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
N_TARGETS = len(TARGETS)
ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
PLANES = ("Sagittal", "Coronal", "Axial")
