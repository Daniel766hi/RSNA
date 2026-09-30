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
args = " ".join(f"--group '{g}/fold*.pt'" for g in groups)
sh(f"python {CODE}/scripts/predict.py --data {DATA} {args} --img {CONFIG['img']} --crop-mm {CONFIG['crop_mm']} "
   + (" --recenter" if CONFIG.get("recenter") else "")
   + f" --budget-hours {CONFIG.get('budget_hours', 8.0)} --workers {os.cpu_count() or 4} --out {WORK}/submission.csv")
