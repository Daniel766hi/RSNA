"""Drive the whole Kaggle pipeline from the command line (Kaggle CLI + KAGGLE_API_TOKEN).

    python kaggle/orchestrate.py whoami
    python kaggle/orchestrate.py upload-code                   # repo -> private dataset <user>/kneemri-code
    python kaggle/orchestrate.py search-labels                 # public report-label datasets to attach
    python kaggle/orchestrate.py push cache
    python kaggle/orchestrate.py push labels --datasets owner/a owner/b
    python kaggle/orchestrate.py push train --slug kneemri-train-r0
    python kaggle/orchestrate.py push train --slug kneemri-train-r1 --refine-from kneemri-train-r0
    python kaggle/orchestrate.py push submit --train kneemri-train-r0
    python kaggle/orchestrate.py wait kneemri-submit
    python kaggle/orchestrate.py submit kneemri-submit -m "r0 convnext-t"

Each stage is one private script kernel; later stages attach earlier kernels' outputs as
``kernel_sources``. Stage configuration is templated into the script (see kaggle/stages/).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "kaggle" / "stages"
BUILD = ROOT / "kaggle" / "build"
COMP = "rsna-knee-abnormality-detection"

# Volume settings shared by the cache and submit stages: they must match (the model was trained
# on this recipe). 320 px keeps the cache under Kaggle's 20 GB output limit.
VOLUME = {"img": 320, "crop_mm": 140.0, "recenter": False}
TRAIN = {"backbone": "convnext_tiny.fb_in22k_ft_in1k_384", "epochs": 10, "batch_size": 4, "grad_accum": 2, "n_folds": 5,
         "max_windows": 16, "grad_ckpt": False}


def run(cmd: list[str], capture: bool = False) -> str:
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        if capture:
            print(r.stdout, r.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout if capture else ""


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username") or getattr(api, "config_values", {}).get("username")
    if not user:
        raise SystemExit("could not determine the Kaggle username; set KAGGLE_USERNAME")
    return user


def upload_code(user: str, message: str) -> None:
    stage = BUILD / "code"
    shutil.rmtree(stage, ignore_errors=True)
    for sub in ("src", "scripts"):
        shutil.copytree(ROOT / sub, stage / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "pyproject.toml", stage / "pyproject.toml")
    meta = {"title": "kneemri-code", "id": f"{user}/kneemri-code", "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/kneemri-code"], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        run(["kaggle", "datasets", "version", "-p", str(stage), "-m", message, "--dir-mode", "zip"])
    else:
        run(["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"])


def build_kernel(user: str, stage: str, slug: str, config: dict, *, gpu: bool, internet: bool,
                 datasets: list[str], kernels: list[str], models: list[str] | None = None) -> Path:
    out = BUILD / slug
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    boot = (STAGES / "_bootstrap.py").read_text().replace("__CONFIG__", json.dumps(config))
    (out / f"{slug}.py").write_text(boot + "\n" + (STAGES / f"{stage}.py").read_text())
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": internet, "dataset_sources": [f"{user}/kneemri-code", *datasets],
        "competition_sources": [COMP], "kernel_sources": [f"{user}/{k}" if "/" not in k else k for k in kernels],
        "model_sources": models or [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return out


def push(user: str, a) -> None:
    if a.stage == "cache":
        folder = build_kernel(user, "cache", a.slug or "kneemri-cache", VOLUME, gpu=False, internet=True,
                              datasets=[], kernels=[])
    elif a.stage == "labels":
        cfg = {"n_teachers": a.n_teachers, "llm_model": a.llm_model}
        folder = build_kernel(user, "labels", a.slug or "kneemri-labels", cfg, gpu=bool(a.llm_model),
                              internet=True, datasets=a.datasets, kernels=[], models=a.models)
    elif a.stage == "train":
        cfg = {**TRAIN, **json.loads(a.config or "{}"), "refine_from": a.refine_from}
        kernels = ["kneemri-cache", "kneemri-labels"] + ([a.refine_from] if a.refine_from else [])
        folder = build_kernel(user, "train", a.slug or "kneemri-train-r0", cfg, gpu=True, internet=True,
                              datasets=[], kernels=kernels)
    elif a.stage == "submit":
        cfg = {**VOLUME, "budget_hours": a.budget_hours}
        folder = build_kernel(user, "submit", a.slug or "kneemri-submit", cfg, gpu=True, internet=False,
                              datasets=[], kernels=["kneemri-cache", *a.train])
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    if a.dry_run:
        print(f"built {folder} (not pushed)")
        return
    run(["kaggle", "kernels", "push", "-p", str(folder)])


def wait(user: str, slug: str, poll: int = 60) -> str:
    ref = slug if "/" in slug else f"{user}/{slug}"
    while True:
        out = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True).stdout.lower()
        status = next((s for s in ("complete", "error", "cancel", "running", "queued") if s in out), "unknown")
        print(time.strftime("%H:%M:%S"), ref, status, flush=True)
        if status in ("complete", "error", "cancel"):
            return status
        time.sleep(poll)


def check_submission(user: str, slug: str) -> None:
    """Refuse to submit a kernel version without a real submission.csv (Kaggle rejects the first
    case, and an all-0.5 fallback file would waste a daily submission)."""
    import pandas as pd  # noqa: PLC0415

    dest = BUILD / "outputs" / slug
    shutil.rmtree(dest, ignore_errors=True)
    run(["kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(dest)])
    sub = dest / "submission.csv"
    if not sub.exists():
        raise SystemExit(f"{slug}: latest version has no submission.csv; not submitting")
    df = pd.read_csv(sub)
    vals = df.drop(columns=df.columns[0])
    if (vals == 0.5).all().all():
        raise SystemExit(f"{slug}: submission.csv is the all-0.5 fallback; check the kernel log")
    print(f"{slug}: submission.csv ok ({len(df)} rows)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    u = sub.add_parser("upload-code")
    u.add_argument("-m", "--message", default="update")
    sub.add_parser("search-labels")
    p = sub.add_parser("push")
    p.add_argument("stage", choices=["cache", "labels", "train", "submit"])
    p.add_argument("--slug")
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--llm-model", default=None)
    p.add_argument("--n-teachers", type=int, default=1)
    p.add_argument("--refine-from", default=None)
    p.add_argument("--config", default=None, help='JSON overrides for TRAIN, e.g. \'{"epochs": 8}\'')
    p.add_argument("--train", nargs="*", default=["kneemri-train-r0"])
    p.add_argument("--budget-hours", type=float, default=8.0)
    p.add_argument("--dry-run", action="store_true")
    w = sub.add_parser("wait")
    w.add_argument("slug")
    o = sub.add_parser("output")
    o.add_argument("slug")
    s = sub.add_parser("submit")
    s.add_argument("slug")
    s.add_argument("-m", "--message", required=True)
    s.add_argument("-v", "--version", default=None)
    a = ap.parse_args()

    user = os.environ.get("KAGGLE_USERNAME") or ("dryrun" if getattr(a, "dry_run", False) else username())
    if a.cmd == "whoami":
        print(user)
        run(["kaggle", "competitions", "list", "-s", "knee"])
    elif a.cmd == "upload-code":
        upload_code(user, a.message)
    elif a.cmd == "search-labels":
        for q in ("rsna knee labels", "knee report labels", "rsna knee soft labels", "knee llm labels"):
            run(["kaggle", "datasets", "list", "-s", q, "--sort-by", "updated"])
    elif a.cmd == "push":
        push(user, a)
    elif a.cmd == "wait":
        status = wait(user, a.slug)
        sys.exit(0 if status == "complete" else 1)
    elif a.cmd == "output":
        dest = BUILD / "outputs" / a.slug
        run(["kaggle", "kernels", "output", f"{user}/{a.slug}", "-p", str(dest)])
    elif a.cmd == "submit":
        check_submission(user, a.slug)
        cmd = ["kaggle", "competitions", "submit", COMP, "-k", f"{user}/{a.slug}", "-f", "submission.csv", "-m", a.message]
        if a.version:
            cmd += ["-v", a.version]
        run(cmd)


if __name__ == "__main__":
    main()
