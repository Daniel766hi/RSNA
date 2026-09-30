"""CSV reading that keeps DICOM UIDs as strings (pandas would parse e.g. ``1.2`` as a float)."""

from __future__ import annotations

import pandas as pd

from .constants import ID_COL, SERIES_COL


def read_table(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={ID_COL: str, SERIES_COL: str})
