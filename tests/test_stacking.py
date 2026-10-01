"""Stacking on synthetic OOF predictions: the rule-gated script runs end to end and never
drops studies or columns from the submission."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kneemri.constants import ID_COL, TARGETS

pytest.importorskip("lightgbm")
ROOT = Path(__file__).resolve().parents[1]


def _write_case(tmp: Path, n: int = 400, n_gold: int = 58, n_test: int = 50) -> dict:
    rng = np.random.default_rng(0)
    ids = [f"1.2.{i}" for i in range(n)]
    latent = rng.normal(size=(n, len(TARGETS)))
    latent[:, 8] += 0.8 * latent[:, 7]  # synovitis co-occurs with effusion
    truth = (latent > 0.5).astype(float)
    train = pd.DataFrame({ID_COL: ids, "Report": "x"})
    for k, t in enumerate(TARGETS):
        train[t] = np.where(np.arange(n) < n_gold, truth[:, k], np.nan)
    (tmp / "data").mkdir()
    train.to_csv(tmp / "data" / "train.csv", index=False)
    teacher = pd.DataFrame(np.clip(truth * 0.8 + rng.uniform(0, 0.2, truth.shape), 0, 1), columns=TARGETS)
    teacher.insert(0, ID_COL, ids)
    teacher.to_csv(tmp / "labels.csv", index=False)
    test_ids = [f"9.9.{i}" for i in range(n_test)]
    pairs = []
    for r in range(2):
        run = tmp / f"run{r}"
        run.mkdir()
        oof = 1 / (1 + np.exp(-(latent + rng.normal(scale=1.0, size=latent.shape))))
        df = pd.DataFrame(oof, columns=TARGETS)
        df.insert(0, ID_COL, ids)
        df.insert(1, "fold", np.arange(n) % 5)
        for k in range(5):
            df[df.fold == k].to_csv(run / f"oof_fold{k}.csv", index=False)
        raw = pd.DataFrame(rng.uniform(size=(n_test, len(TARGETS))), columns=TARGETS)
        raw.insert(0, ID_COL, test_ids)
        raw.to_csv(tmp / f"raw_g{r}.csv", index=False)
        pairs += ["--pair", str(run), str(tmp / f"raw_g{r}.csv")]
    base = pd.DataFrame(rng.uniform(size=(n_test, len(TARGETS))), columns=TARGETS)
    base.insert(0, ID_COL, test_ids)
    base.to_csv(tmp / "submission.csv", index=False)
    return {"pairs": pairs, "test_ids": test_ids}


@pytest.mark.parametrize("min_gain", [-1.0, 1.0])  # forced use / forced fallback
def test_stack_script(tmp_path: Path, min_gain: float) -> None:
    case = _write_case(tmp_path)
    before = pd.read_csv(tmp_path / "submission.csv")
    cmd = [sys.executable, str(ROOT / "scripts" / "stack.py"), "--data", str(tmp_path / "data"),
           "--labels", str(tmp_path / "labels.csv"), *case["pairs"], "--baseline", str(tmp_path / "submission.csv"),
           "--out", str(tmp_path / "submission.csv"), "--report", str(tmp_path / "rep.json"),
           "--min-gain", str(min_gain)]
    subprocess.run(cmd, check=True)
    rep = json.loads((tmp_path / "rep.json").read_text())
    after = pd.read_csv(tmp_path / "submission.csv")
    assert rep["n_gold"] == 58 and rep["n_runs"] == 2
    assert after.columns.tolist() == [ID_COL] + TARGETS
    assert after[ID_COL].astype(str).tolist() == case["test_ids"]
    assert after[TARGETS].notna().all().all()
    if min_gain > 0:
        assert not rep["use"]
        pd.testing.assert_frame_equal(after, before)
    else:
        assert rep["use"] == (rep["weak_stack"] >= rep["weak_base"])
