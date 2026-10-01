# Stage 3 (2xT4, internet on for timm weights): cross-validated training on the cache.
# Folds are split across the two GPUs as two parallel processes. With CONFIG["refine_from"]
# set, labels are first refined with that run's out-of-fold predictions (lever L1).
from kneemri.kaggle_env import find_competition_dir, find_files, find_npz_cache, find_npz_caches

DATA = find_competition_dir()
CACHE_DIRS = find_npz_caches(CONFIG["caches"]) if CONFIG.get("caches") else [find_npz_cache()]
CACHE = CACHE_DIRS[0]
print("cache dirs:", [str(d) for d in CACHE_DIRS], flush=True)
labels = [p for p in find_files("labels.csv") if str(WORK) not in str(p)][0]
work = WORK
if CONFIG.get("refine_from"):
    oof = [p for p in find_files("oof_fold*.csv") if CONFIG["refine_from"] in str(p)]
    if not oof:
        raise SystemExit(f"no OOF files from {CONFIG['refine_from']}")
    sh(f"python {CODE}/scripts/refine_labels.py --data {DATA} --labels {labels} "
       f"--oof '{oof[0].parent}/oof_fold*.csv' --out {work}/labels_refined.csv")
    labels = work / "labels_refined.csv"

n_folds = CONFIG["n_folds"]
import torch  # noqa: E402

n_gpu = max(1, torch.cuda.device_count())
common = (f"python {CODE}/scripts/train_cv.py --data {DATA} --cache {",".join(map(str, CACHE_DIRS))} --labels {labels} --out {work}/run "
          f"--backbone {CONFIG['backbone']} --epochs {CONFIG['epochs']} --batch-size {CONFIG['batch_size']} --grad-accum {CONFIG.get('grad_accum', 1)} "
          f"--n-folds {n_folds} --workers {max(1, (os.cpu_count() or 4) // n_gpu)} --max-windows {CONFIG['max_windows']}"
          + (" --grad-ckpt" if CONFIG.get("grad_ckpt") else "")
          + ("" if CONFIG.get("pretrained", True) else " --no-pretrained")
          + (f" --img-size {CONFIG['img_size']}" if CONFIG.get("img_size") else ""))
procs = []
for g in range(n_gpu):
    folds = [k for k in range(n_folds) if k % n_gpu == g]
    if not folds:
        continue
    cmd = f"PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES={g} {common} --folds {' '.join(map(str, folds))} > {work}/train_gpu{g}.log 2>&1"
    print("$", cmd, flush=True)
    procs.append(subprocess.Popen(cmd, shell=True))
codes = [p.wait() for p in procs]
for g in range(n_gpu):
    print(f"==== gpu{g} log tail ====\n" + "".join(open(f"{work}/train_gpu{g}.log").readlines()[-40:]), flush=True)
if any(codes):
    raise SystemExit(f"training failed: {codes}")
if (CACHE / "volume.json").exists():  # tells the submit stage which volume recipe this run needs
    import shutil
    shutil.copy(CACHE / "volume.json", work / "run" / "volume.json")
print(f"done in {(time.time() - T0) / 3600:.2f} h", flush=True)
