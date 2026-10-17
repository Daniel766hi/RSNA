# Reader stage 4 (CPU): compare the retrained reader with the published one on the 58 gold
# studies, the way the 0.954 notebook uses it (sigmoid mean of the fold checkpoints).
#   published v0 : kneemri-reader-v0eval (cnxt_v0_fold0-2 scored on our cache)
#   l1           : our retrain on L1-refined labels
#   v0 T4 port   : the author's labels through our T4 port (separates labels from the port)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kneemri.constants import TARGETS

lab = next(p for p in INPUT.rglob("labels_v0.csv") if "/kneemri-rlabels/" in str(p))
gold = pd.read_csv(lab, index_col=0)
gold = gold[gold.is_gold == 1][[f"gold_{t}" for t in TARGETS]].set_axis(TARGETS, axis=1)


def load(kernel, sub="run"):
    files = sorted(p for p in INPUT.rglob("oof_fold*.csv") if f"/{kernel}/{sub}/" in str(p))
    out = {}
    for f in files:
        df = pd.read_csv(f, index_col=0)
        g = df[df.set == "gold"][TARGETS]
        out[int(f.stem[8:])] = 1 / (1 + np.exp(-g))
    return out


def macro(p):
    p = p.reindex(gold.index)
    return float(np.mean([roc_auc_score(gold[t], p[t]) for t in TARGETS]))


arms = {"published v0": load("kneemri-reader-v0eval"),
        "l1": {**load("kneemri-reader-l1a"), **load("kneemri-reader-l1b")},
        "v0 T4 port": load("kneemri-reader-l1b", "run_v0")}
rows = []
for name, folds in arms.items():
    for k, p in sorted(folds.items()):
        rows.append({"arm": name, "folds": str(k), "gold_macro": macro(p)})
    common = sorted(set(folds) & {0, 1, 2})
    if len(common) > 1:
        rows.append({"arm": name, "folds": "mean of " + ",".join(map(str, common)),
                     "gold_macro": macro(sum(folds[k] for k in common) / len(common))})
pub, l1 = arms["published v0"], arms["l1"]
common = sorted(set(pub) & set(l1))
if common:  # the 'add' option: published and retrained checkpoints side by side
    both = (sum(pub[k] for k in common) + sum(l1[k] for k in common)) / (2 * len(common))
    rows.append({"arm": "published v0 + l1", "folds": ",".join(map(str, common)), "gold_macro": macro(both)})
    per = pd.DataFrame({"published v0": [roc_auc_score(gold[t], (sum(pub[k] for k in common) / len(common))
                                                       .reindex(gold.index)[t]) for t in TARGETS],
                        "l1": [roc_auc_score(gold[t], (sum(l1[k] for k in common) / len(common))
                                             .reindex(gold.index)[t]) for t in TARGETS]}, index=TARGETS)
    print("[reader_compare] per-target gold AUC (fold mean):\n" + per.round(3).to_string(), flush=True)
rep = pd.DataFrame(rows)
print("[reader_compare]\n" + rep.round(4).to_string(index=False), flush=True)
rep.to_csv(WORK / "reader_compare.csv", index=False)
