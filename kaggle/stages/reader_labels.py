# Reader stage 2 (CPU): training labels for the retrained 2.5-D ConvNeXt reader.
#   labels_v0.csv : the reader author's own recipe, unchanged (make_labels.py v0: mean of four
#                   public LLM report-label tables, report-grouped 5-fold split, gold kept apart).
#   labels_l1.csv : the same table and folds, with every cell refined by our lever L1 (calibrated
#                   OOF image predictions of our KneeMIL runs, weighted towards the image where the
#                   v0 consensus is unsure). Only the targets change, so a reader trained on it is a
#                   controlled test of our labels against the published v0 reader.
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kneemri.constants import ID_COL, TARGETS
from kneemri.ensemble import rank_mean
from kneemri.kaggle_env import find_competition_dir, find_files
from kneemri.labels.refine import explicitness_from_soft, refine_labels

DATA = find_competition_dir()
SRC = next(p.parent for p in sorted(INPUT.rglob("make_labels.py")) if (p.parent / "knee.py").exists())
OUT = WORK / "reader_labels"
EXT = WORK / "ext"
OUT.mkdir(exist_ok=True)
EXT.mkdir(exist_ok=True)
# make_labels.py expects <EXT>/<owner>_<slug>/<file>; link the attached mounts under those names
for name, fname in {"stevenleehans_rsna-knee-llm-report-labels": "llm_labels_v4_blend.csv",
                    "pilkwang_rsna-knee-llm-labels": "report_labels_v2.csv",
                    "lixin73_rsna-knee-llm-report-labels-sol56": "labels_llm_gpt56sol.csv",
                    "laymond_rsna-knee-abnormality-qwen3-8b-weak-labels": "qwen_knee_weak_labels.csv"}.items():
    src = next(p for p in INPUT.rglob(fname) if str(WORK) not in str(p))
    if not (EXT / name).exists():
        (EXT / name).symlink_to(src.parent, target_is_directory=True)

spec = importlib.util.spec_from_file_location("reader_make_labels", SRC / "make_labels.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)
ml.DATA, ml.EXT, ml.WORK = str(DATA), str(EXT), str(OUT)
assert ml.LABELS == TARGETS
sys.argv = ["make_labels.py", "v0"]
ml.main()

v0 = pd.read_csv(OUT / "labels_v0.csv", index_col=0)
oofs = []
for r in CONFIG["refine_from"]:
    files = sorted(p for p in find_files("oof_fold*.csv") if f"/{r}/" in str(p))
    if not files:
        raise SystemExit(f"no OOF files from {r}")
    run = pd.concat([pd.read_csv(f, dtype={ID_COL: str}) for f in files]).dropna(subset=TARGETS)
    oofs.append(run.drop_duplicates(ID_COL, keep="last"))
    print(f"teacher {r}: {len(oofs[-1])} OOF studies", flush=True)
oof = rank_mean(oofs).set_index(ID_COL)[TARGETS]
y = v0[TARGETS].astype(float)
refined = refine_labels(y, explicitness_from_soft(y), oof, CONFIG.get("w_explicit", 0.85),
                        CONFIG.get("w_silent", 0.25), gold=None).clip(0.01, 0.99)
l1 = v0.copy()
l1[TARGETS] = refined[TARGETS]
l1.to_csv(OUT / "labels_l1.csv")

# diagnostic only: agreement of each target table with the official labels on the gold rows
# (the OOF used for a gold row comes from models that did not train on it)
g = v0[v0.is_gold == 1]
rep = pd.DataFrame({name: [roc_auc_score(g[f"gold_{t}"], tab.loc[g.index, t]) for t in TARGETS]
                    for name, tab in {"v0": v0, "l1": l1, "oof": oof.reindex(v0.index).fillna(0.5)}.items()},
                   index=TARGETS)
rep.loc["MEAN"] = rep.mean()
print("[reader_labels] gold agreement of the target tables:\n" + rep.round(3).to_string(), flush=True)
print("[reader_labels] mean |l1 - v0| per target:\n"
      + (l1.loc[v0.is_gold == 0, TARGETS] - v0.loc[v0.is_gold == 0, TARGETS]).abs().mean().round(4).to_string(),
      flush=True)
