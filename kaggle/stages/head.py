# Stage 3b (2xT4): retrain each fold's MIL head on full-study context over the frozen stage-1
# backbone (scripts/train_head.py). Labels are built as in the train stage (teacher table, or
# refined with CONFIG["refine_from"]'s OOF). Folds are split over the GPUs.
from kneemri.kaggle_env import find_competition_dir, find_files, find_npz_cache, find_npz_caches

DATA = find_competition_dir()
CACHE_DIRS = find_npz_caches(CONFIG["caches"]) if CONFIG.get("caches") else [find_npz_cache()]
print("cache dirs:", [str(d) for d in CACHE_DIRS], flush=True)
run_dir = next(p.parent for p in find_files("fold0.pt") if CONFIG["run"] in str(p))
labels = [p for p in find_files("labels.csv") if "kneemri-labels" in str(p)][0]
work = WORK
if CONFIG.get("refine_from"):
    oof = [p for p in find_files("oof_fold*.csv") if CONFIG["refine_from"] in str(p)]
    if not oof:
        raise SystemExit(f"no OOF files from {CONFIG['refine_from']}")
    sh(f"python {CODE}/scripts/refine_labels.py --data {DATA} --labels {labels} "
       f"--oof '{oof[0].parent}/oof_fold*.csv' --out {work}/labels_refined.csv")
    labels = work / "labels_refined.csv"

import torch  # noqa: E402

n_gpu = max(1, torch.cuda.device_count())
folds_all = sorted(int(p.stem[4:]) for p in run_dir.glob("fold*.pt") if p.stem[4:].isdigit())
common = (f"python {CODE}/scripts/train_head.py --data {DATA} --cache {','.join(map(str, CACHE_DIRS))} "
          f"--labels {labels} --run {run_dir} --out {work}/run --views {CONFIG['views']} "
          f"--epochs {CONFIG['epochs']} --seeds {CONFIG['seeds']} --workers {max(1, (os.cpu_count() or 4) // n_gpu)}")
procs = []
for g in range(n_gpu):
    folds = [k for i, k in enumerate(folds_all) if i % n_gpu == g]
    if not folds:
        continue
    cmd = f"CUDA_VISIBLE_DEVICES={g} {common} --folds {' '.join(map(str, folds))} > {work}/head_gpu{g}.log 2>&1"
    print("$", cmd, flush=True)
    procs.append(subprocess.Popen(cmd, shell=True))
codes = [p.wait() for p in procs]
for g in range(n_gpu):
    print(f"==== gpu{g} log tail ====\n" + "".join(open(f"{work}/head_gpu{g}.log").readlines()[-45:]), flush=True)
if any(codes):
    raise SystemExit(f"head training failed: {codes}")
print(f"done in {(time.time() - T0) / 3600:.2f} h", flush=True)
