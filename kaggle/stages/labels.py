# Stage 2: build the training-target table.
# Every attached CSV that looks like a report-label table is leak-checked against the gold
# rows; leaky tables are dropped, and the remaining ones are ranked by gold agreement. By
# default the single best clean table is used (the public study found that fusing tables of
# shared lineage does not beat the best one). With CONFIG["llm_model"] set, an offline LLM
# labeler run is added as another teacher.
import pandas as pd

from kneemri.constants import ID_COL, TARGETS
from kneemri.io import read_table
from kneemri.kaggle_env import find_competition_dir, find_files
from kneemri.labels.teachers import detect_gold_leak, fuse_teachers, gold_agreement, gold_rows, normalise_columns

DATA = find_competition_dir()
train = read_table(DATA / "train.csv")
gold = gold_rows(train)
out = WORK

cands = {}
for p in find_files("*.csv"):
    if DATA in p.parents or p.name in ("train.csv", "test.csv", "sample_submission.csv"):
        continue
    try:
        df = normalise_columns(read_table(p))
    except Exception:
        continue
    if ID_COL not in df or sum(t in df for t in TARGETS) < 12:
        continue
    df = df[[ID_COL] + TARGETS].drop_duplicates(ID_COL).set_index(ID_COL)
    df = df.apply(pd.to_numeric, errors="coerce").clip(0, 1)
    if df.index.isin(train[ID_COL]).mean() < 0.5 or len(df) < 0.8 * len(train):
        continue
    if df.isna().to_numpy().mean() > 0.2:  # mostly-empty tables cannot serve as targets
        print(f"skip {p}: {df.isna().to_numpy().mean():.0%} empty cells", flush=True)
        continue
    cands[str(p)] = df

if CONFIG.get("llm_model"):
    sh(f"pip install -q vllm", check=False)
    llm = list(find_files("config.json", INPUT))
    model_dir = next((str(p.parent) for p in llm if CONFIG["llm_model"] in str(p)), None)
    if not model_dir:
        raise SystemExit(f"LLM {CONFIG['llm_model']} not attached (add it with --models)")
    sh(f"python {CODE}/scripts/label_reports_llm.py --data {DATA} --model {model_dir} --out {out}/llm_labels.csv")
    cands["llm"] = normalise_columns(read_table(out / "llm_labels.csv")).set_index(ID_COL)[TARGETS]

rows = []
for name, df in cands.items():
    leak = detect_gold_leak(df, gold)
    agree = gold_agreement(df, gold) if not leak["leak"] else None
    rows.append({"source": name, "n": len(df), "nan_frac": round(float(df.isna().to_numpy().mean()), 3), "leak": leak["leak"], "exact_match": round(leak["exact_match"], 3),
                 "gold_macro": None if agree is None else round(float(agree.mean()), 4)})
report = pd.DataFrame(rows).sort_values("gold_macro", ascending=False, na_position="last")
print(report.to_string(index=False), flush=True)
report.to_csv(out / "teacher_report.csv", index=False)
clean = [r["source"] for _, r in report.iterrows() if not r["leak"] and r["gold_macro"] is not None]
if not clean:
    raise SystemExit("no clean teacher table found: attach public label datasets or set llm_model")
chosen = clean[: CONFIG.get("n_teachers", 1)]
if CONFIG.get("force_llm"):  # a new, decorrelated label source on purpose, whatever its rank
    if "llm" not in clean:
        raise SystemExit("force_llm: the LLM table is missing or flagged as leaky")
    chosen = ["llm"]
print("using:", chosen, flush=True)
fused = fuse_teachers([cands[c] for c in chosen])
fused = fused.reindex(train[ID_COL])
fused.index.name = ID_COL
if "llm" in chosen:  # keep the state-aware weights / explicitness for L1
    llm = read_table(out / "llm_labels.csv").set_index(ID_COL)
    for t in TARGETS:
        fused[f"{t}__e"] = llm[f"{t}__e"].reindex(fused.index)
        if len(chosen) == 1:  # silent cells train at low weight (llm_labeler.to_soft_labels)
            fused[f"{t}__w"] = llm[f"{t}__w"].reindex(fused.index)
fused.reset_index().to_csv(out / "labels.csv", index=False)
print("gold agreement of the chosen labels:\n", gold_agreement(fused[TARGETS], gold).round(3).to_string())
