# ---- shared bootstrap, prepended to every stage script by kaggle/orchestrate.py ----
import json, os, subprocess, sys, time
from pathlib import Path

CONFIG = json.loads(r'''__CONFIG__''')
T0 = time.time()


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


KROOT = Path(os.environ.get("KNEEMRI_KAGGLE_ROOT", "/kaggle"))
INPUT, WORK = KROOT / "input", KROOT / "working"


def find_code_root():
    for p in sorted(INPUT.rglob("kneemri/__init__.py")):
        if p.parent.parent.name == "src":
            return p.parent.parent.parent
    raise SystemExit("kneemri code dataset not attached")


CODE = find_code_root()
sys.path.insert(0, str(CODE / "src"))
os.environ["PYTHONPATH"] = str(CODE / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
print("config:", json.dumps(CONFIG), "| code:", CODE, flush=True)
# ---- end bootstrap ----
