import numpy as np
import pandas as pd

from kneemri.constants import ID_COL, TARGETS
from kneemri.labels.llm_labeler import build_messages, parse_response, to_soft_labels
from kneemri.labels.refine import refine_labels
from kneemri.labels.teachers import detect_gold_leak, fuse_teachers, gold_agreement, normalise_columns


def _frame(values, ids):
    return pd.DataFrame(values, index=pd.Index(ids, name=ID_COL), columns=TARGETS)


def test_parse_response_tolerates_fences_and_missing_keys():
    txt = '```json\n{"ACL": {"state": "present", "confidence": 0.9}, "medial meniscus": {"state": "Absent", "confidence": 1}}\n```'
    p = parse_response(txt)
    assert p["ACL"] == ("present", 0.9)
    assert p["Medial Meniscus"] == ("absent", 1.0)
    assert p["Synovitis"][0] == "not_mentioned"
    assert all(t in p for t in TARGETS)
    assert parse_response("garbage")["ACL"][0] == "not_mentioned"


def test_soft_labels_keep_silence_apart_from_negation():
    parsed = {t: ("not_mentioned", 0.5) for t in TARGETS}
    parsed["ACL"] = ("present", 1.0)
    parsed["MCL"] = ("absent", 1.0)
    y, w, e = to_soft_labels(parsed)
    assert y["ACL"] == 1.0 and y["MCL"] == 0.0 and e["ACL"] == e["MCL"] == 1.0
    assert e["Synovitis"] == 0.0 and w["Synovitis"] < w["ACL"]


def test_prompt_mentions_every_target():
    msgs = build_messages("Rotura del LCA.")
    assert all(t in msgs[1]["content"] for t in TARGETS)


def test_normalise_columns():
    df = pd.DataFrame(columns=["studyinstanceuid", "Bakers", "medial_meniscus", "pf_oa"])
    assert list(normalise_columns(df).columns) == [ID_COL, "Baker's", "Medial Meniscus", "PF OA"]


def test_gold_leak_detection_and_agreement():
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(40)]
    gold = _frame(rng.integers(0, 2, (40, 12)).astype(float), ids)
    leaky = gold.copy()
    honest = _frame(rng.random((40, 12)), ids)
    assert detect_gold_leak(leaky, gold)["leak"]
    assert not detect_gold_leak(honest, gold)["leak"]
    assert gold_agreement(leaky, gold).min() == 1.0


def test_fuse_teachers_handles_partial_coverage():
    a = _frame(np.full((2, 12), 0.2), ["x", "y"])
    b = _frame(np.full((1, 12), 0.8), ["y"])
    f = fuse_teachers([a, b])
    assert np.allclose(f.loc["x"], 0.2) and np.allclose(f.loc["y"], 0.5)


def test_refine_moves_silent_cells_towards_image_and_keeps_gold():
    rng = np.random.default_rng(1)
    ids = [f"s{i}" for i in range(200)]
    truth = rng.integers(0, 2, (200, 12)).astype(float)
    y = _frame(np.where(truth > 0, 0.95, 0.05), ids)
    explicit = _frame(np.ones((200, 12)), ids)
    y.iloc[:50, 8], explicit.iloc[:50, 8] = 0.1, 0.0          # silent synovitis cells
    oof = _frame(np.clip(truth * 0.7 + rng.random((200, 12)) * 0.3, 0, 1), ids)
    gold = _frame(np.ones((1, 12)), ["s0"])
    out = refine_labels(y, explicit, oof, gold=gold)
    silent_pos = [i for i in range(1, 50) if truth[i, 8] == 1]
    assert (out.iloc[silent_pos, 8] > y.iloc[silent_pos, 8]).all()
    assert (out.loc["s0"] == 1.0).all()
    explicit_rows = out.iloc[50:, 0] - y.iloc[50:, 0]
    assert explicit_rows.abs().max() < 0.2                  # explicit cells barely move
