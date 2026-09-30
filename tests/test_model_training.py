import numpy as np
import pandas as pd
import torch

from conftest import make_study
from kneemri.constants import ID_COL, TARGETS
from kneemri.data import KneeStudyDataset, collate, study_to_windows
from kneemri.ensemble import rank_mean
from kneemri.infer import predict_dicom
from kneemri.losses import composite_loss, pairwise_auc_loss
from kneemri.metrics import macro_auc, paired_bootstrap
from kneemri.model import KneeMIL, ModelConfig
from kneemri.train import TrainConfig, build_targets, make_folds, run_cv
from kneemri.volume import VolumeConfig, build_study, save_study

TINY = dict(backbone="resnet18", pretrained=False, d_model=32, n_heads=4)


def _arrays(n_slots=5, n=4, img=16, seed=0):
    rng = np.random.default_rng(seed)
    names = ["sag_fs", "sag_anat", "cor_fs", "cor_anat", "ax_fs"][:n_slots]
    out = {}
    for s in names:
        out[f"{s}_img"] = rng.integers(0, 255, (n, img, img), dtype=np.uint8)
        out[f"{s}_pos"] = np.linspace(0, 1, n).astype(np.float32)
    return out


def test_windows_are_triplets_of_neighbours():
    a = _arrays(n_slots=2, n=3)
    win, sid, pos = study_to_windows(a)
    assert win.shape == (6, 3, 16, 16) and list(sid) == [0, 0, 0, 1, 1, 1]
    assert np.array_equal(win[1, 0], a["sag_fs_img"][0]) and np.array_equal(win[1, 2], a["sag_fs_img"][2])
    assert np.array_equal(win[0, 0], win[0, 1])        # clamped at the edge


def test_collate_pads_and_masks():
    ds = KneeStudyDataset([""] * 2)
    b = collate([ds.from_arrays(_arrays(n=4), 0), ds.from_arrays(_arrays(n_slots=2, n=4), 1)])
    assert b["x"].shape[:2] == (2, 20) and b["mask"].sum(1).tolist() == [20, 8]


def test_model_forward_and_position_sensitivity():
    torch.manual_seed(0)
    ds = KneeStudyDataset([""] * 2)
    b = collate([ds.from_arrays(_arrays(img=32), 0), ds.from_arrays(_arrays(n_slots=3, img=32, seed=1), 1)])
    for use_pos in (True, False):
        m = KneeMIL(ModelConfig(**TINY, use_position=use_pos)).eval()
        with torch.no_grad():
            out = m(b["x"], b["slot"], b["pos"], b["mask"])
            out2 = m(b["x"], b["slot"], b["pos"].flip(1), b["mask"])
            perm = torch.randperm(20)
            out3 = m(b["x"][:1, perm], b["slot"][:1, perm], b["pos"][:1, perm], b["mask"][:1, perm])
        assert out.shape == (2, len(TARGETS)) and torch.isfinite(out).all()
        assert torch.allclose(out[:1], out3, atol=1e-4)        # (window, position) pairs are a set
        assert (not torch.allclose(out, out2, atol=1e-4)) == use_pos


def test_losses_and_metrics():
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    w = torch.ones_like(y)
    good = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
    assert pairwise_auc_loss(good, y, w).item() == 0.0
    assert composite_loss(-good, y, w).item() > composite_loss(good, y, w).item()
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, (60, 12)).astype(float)
    p_good = yy * 0.6 + rng.random((60, 12)) * 0.4
    assert macro_auc(yy, p_good) > 0.9
    bs = paired_bootstrap(yy, rng.random((60, 12)), p_good, n=200)
    assert bs["delta"] > 0.3 and bs["p_gt_0"] > 0.95


def test_rank_mean_is_rank_based():
    a = pd.DataFrame({ID_COL: ["a", "b", "c"], **{t: [0.1, 0.2, 0.3] for t in TARGETS}})
    b = pd.DataFrame({ID_COL: ["c", "b", "a"], **{t: [300.0, 200.0, 100.0] for t in TARGETS}})
    r = rank_mean([a, b])
    assert list(r["ACL"]) == sorted(r["ACL"])


def test_folds_group_identical_reports_and_spread_gold():
    n = 40
    train = pd.DataFrame({ID_COL: [f"s{i}" for i in range(n)], "Report": [f"r{i // 2}" for i in range(n)]})
    for t in TARGETS:
        train[t] = [1.0 if i < 10 else np.nan for i in range(n)]
    f = make_folds(train, 5, seed=0)
    for i in range(0, n, 2):
        assert f[f"s{i}"] == f[f"s{i + 1}"]
    gold_per_fold = f.iloc[:10].value_counts()
    assert gold_per_fold.max() - gold_per_fold.min() <= 2
    y, w, gold = build_targets(train, pd.DataFrame(0.3, index=train[ID_COL], columns=TARGETS), gold_weight=3.0)
    assert gold.sum() == 10 and (w[gold] == 3.0).all() and np.allclose(y[~gold], 0.3)


def test_end_to_end_train_and_predict(tmp_path):
    """Synthetic DICOM -> cache -> 2-fold CV (1 epoch) -> checkpoints -> DICOM inference."""
    vol = VolumeConfig(img=32, crop_mm=60)
    series, train_rows = [], []
    for i in range(8):
        sid = f"9.{i}"
        s = make_study(tmp_path / "dcm", sid, laterality="L" if i % 2 else "R", seed=i)
        series.append(s)
        arrays, meta = build_study(tmp_path / "dcm" / sid, s, vol)
        save_study(tmp_path / "cache" / f"{sid}.npz", arrays, meta)
        train_rows.append({ID_COL: sid, "Report": f"report {i}", **{t: (float(i % 2) if i < 4 else np.nan) for t in TARGETS}})
    train = pd.DataFrame(train_rows)
    labels = pd.DataFrame(np.tile([[0.2], [0.8]], (4, 12)), index=train[ID_COL], columns=TARGETS)
    cfg = TrainConfig(epochs=1, batch_size=2, n_folds=2, num_workers=0, amp=False, out_dir=str(tmp_path / "run"),
                      max_windows=12, model=ModelConfig(**TINY))
    oof = run_cv(train, tmp_path / "cache", labels, None, cfg)
    assert oof[TARGETS].notna().all().all() and (tmp_path / "run" / "metrics.json").exists()

    ckpts = sorted((tmp_path / "run").glob("fold*.pt"))
    pred = predict_dicom([f"9.{i}" for i in range(3)] + ["missing"], pd.concat(series), tmp_path / "dcm", ckpts, vol,
                         num_workers=0)
    assert pred.shape == (4, 13) and ((pred[TARGETS] >= 0) & (pred[TARGETS] <= 1)).all().all()
    assert (pred.iloc[3][TARGETS] == 0.5).all()                 # undecodable study -> 0.5, no crash
