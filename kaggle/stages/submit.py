# Stage 4 (2xT4, internet OFF): DICOM -> submission.csv. Install the codec wheels from the cache
# kernel's output, then run every checkpoint group attached (one group per training kernel).
from kneemri.kaggle_env import find_competition_dir, find_files

DATA = find_competition_dir()
wheels = find_files("pylibjpeg*.whl")
if wheels:
    sh(f"pip install -q --no-index --find-links {wheels[0].parent} pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg",
       check=False)
groups = sorted({str(p.parent) for p in find_files("fold*.pt")})
print("checkpoint groups:", groups, flush=True)
def _spec(g):
    v = Path(g) / "volume.json"  # runs trained on a non-default cache record its recipe
    return "@{img}:{crop_mm}".format(**json.loads(v.read_text())) if v.exists() else ""


args = " ".join(f"--group '{g}/fold*.pt{_spec(g)}'" for g in groups)
sh(f"python {CODE}/scripts/predict.py --data {DATA} {args} --img {CONFIG['img']} --crop-mm {CONFIG['crop_mm']} "
   + (" --recenter" if CONFIG.get("recenter") else "")
   + f" --budget-hours {CONFIG.get('budget_hours', 8.0)} --workers {os.cpu_count() or 4} --out {WORK}/submission.csv"
   + f" --raw-dir {WORK}/raw",
   check=False)
# Optional second level (pre-registered rule inside scripts/stack.py; keeps the baseline otherwise).
teacher = [p for p in find_files("labels.csv") if "kneemri-labels" in str(p)]
raws = [WORK / "raw" / f"raw_g{i}.csv" for i in range(len(groups))]
if CONFIG.get("stack") and teacher and (WORK / "submission.csv").exists() and all(r.exists() for r in raws):
    pairs = " ".join(f"--pair '{g}' '{r}'" for g, r in zip(groups, raws))
    sh(f"python {CODE}/scripts/stack.py --data {DATA} --labels {teacher[0]} {pairs} "
       f"--baseline {WORK}/submission.csv --out {WORK}/submission.csv --report {WORK}/stack_report.json",
       check=False)
# Kaggle rejects a submission whose notebook version has no submission.csv, so always leave one.
if not (WORK / "submission.csv").exists():
    import shutil
    print("WARNING: predict.py wrote no submission.csv; writing the 0.5 sample instead", flush=True)
    shutil.copy(DATA / "sample_submission.csv", WORK / "submission.csv")
