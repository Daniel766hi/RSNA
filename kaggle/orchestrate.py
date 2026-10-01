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
import re
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


PUBLIC_BASE = "yamadan96/rsna-knee-d4-public0946"  # public 0.943 pipeline (lever L6 partner)


def build_blend(user: str, slug: str, base: str, train: list[str], weight: float) -> Path:
    """Fork a public submission notebook and append a cell that rank-blends our runs into its
    submission.csv at ``weight``. The public pipeline's own output is kept if our part fails."""
    out = BUILD / slug
    shutil.rmtree(out, ignore_errors=True)
    pulled = BUILD / "public" / base.replace("/", "__")
    shutil.rmtree(pulled, ignore_errors=True)
    run(["kaggle", "kernels", "pull", base, "-p", str(pulled), "-m"])
    src_meta = json.loads((pulled / "kernel-metadata.json").read_text())
    nb = json.loads((pulled / src_meta["code_file"]).read_text())
    cfg = {**VOLUME, "weight": weight, "max_hours": 2.5, "total_hours": 8.6}
    cell = (STAGES / "blend_cell.py").read_text().replace("__CONFIG__", json.dumps(cfg))
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": cell})
    out.mkdir(parents=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb))
    keep = ("dataset_sources", "kernel_sources", "competition_sources", "model_sources", "docker_image",
            "machine_shape", "enable_gpu")
    meta = {k: src_meta[k] for k in keep if k in src_meta}
    meta["dataset_sources"] = [d for d in meta.get("dataset_sources", []) if d] + [f"{user}/kneemri-code"]
    meta["kernel_sources"] = meta.get("kernel_sources", []) + [f"{user}/kneemri-cache"] + [
        f"{user}/{k}" if "/" not in k else k for k in train]
    meta.update({"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
                 "kernel_type": "notebook", "is_private": True, "enable_internet": False, "enable_tpu": False})
    if meta.get("docker_image"):
        meta["docker_image_pinning_type"] = "original"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return out


def push(user: str, a) -> None:
    if a.stage == "cache":
        vol = {**VOLUME, **({"img": a.img} if a.img else {}), **({"crop_mm": a.crop_mm} if a.crop_mm else {}),
               **({"shard": a.shard} if a.shard else {})}
        folder = build_kernel(user, "cache", a.slug or "kneemri-cache", vol, gpu=False, internet=True,
                              datasets=[], kernels=[])
    elif a.stage == "labels":
        cfg = {"n_teachers": a.n_teachers, "llm_model": a.llm_model}
        folder = build_kernel(user, "labels", a.slug or "kneemri-labels", cfg, gpu=bool(a.llm_model),
                              internet=True, datasets=a.datasets, kernels=[], models=a.models)
    elif a.stage == "train":
        cfg = {**TRAIN, **json.loads(a.config or "{}"), "refine_from": a.refine_from,
               "caches": a.cache if a.cache != ["kneemri-cache"] else None}
        kernels = [*a.cache, "kneemri-labels"] + ([a.refine_from] if a.refine_from else [])
        folder = build_kernel(user, "train", a.slug or "kneemri-train-r0", cfg, gpu=True, internet=True,
                              datasets=[], kernels=kernels)
    elif a.stage == "submit":
        cfg = {**VOLUME, "budget_hours": a.budget_hours, "stack": a.stack}
        folder = build_kernel(user, "submit", a.slug or "kneemri-submit", cfg, gpu=True, internet=False,
                              datasets=[], kernels=["kneemri-cache", *a.train] + (["kneemri-labels"] if a.stack else []))
    elif a.stage == "head":
        if not a.run:
            raise SystemExit("push head needs --run <stage-1 train kernel>")
        cfg = {"run": a.run, "caches": a.cache if a.cache != ["kneemri-cache"] else None,
               "refine_from": a.refine_from, "views": a.views, "epochs": a.head_epochs, "seeds": a.seeds}
        kernels = [*a.cache, "kneemri-labels", a.run] + ([a.refine_from] if a.refine_from and a.refine_from != a.run else [])
        folder = build_kernel(user, "head", a.slug or f"{a.run}-head", cfg, gpu=True, internet=True,
                              datasets=[], kernels=kernels)
    elif a.stage == "blend":
        folder = build_blend(user, a.slug or "kneemri-blend", a.base, a.train, a.weight)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    if a.dry_run:
        print(f"built {folder} (not pushed)")
        return
    out = run(["kaggle", "kernels", "push", "-p", str(folder)], capture=True)
    print(out, flush=True)
    m = re.search(r"Kernel version (\d+)", out)
    if m:  # code-competition submissions must name the kernel version
        (BUILD / f"{json.loads((folder / 'kernel-metadata.json').read_text())['title']}.version").write_text(m.group(1))


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
    case, and an all-0.5 fallback file would waste a daily submission).

    Uses the file listing and the log, not a download: output downloads come from
    kaggleusercontent.com, which a restricted network may block.
    """
    ref = f"{user}/{slug}"
    files = run(["kaggle", "kernels", "files", ref], capture=True)
    if not any(line.split()[:1] == ["submission.csv"] for line in files.splitlines()):
        raise SystemExit(f"{slug}: latest version has no submission.csv; not submitting")
    log = "".join(e.get("data", "") for e in json.loads(run(["kaggle", "kernels", "logs", ref], capture=True)))
    if "writing the 0.5 sample instead" in log or "[predict] wrote" not in log:
        raise SystemExit(f"{slug}: submission.csv is the 0.5 fallback or predict.py did not finish; check the log")
    print(next(line for line in log.splitlines() if "[predict] wrote" in line), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    u = sub.add_parser("upload-code")
    u.add_argument("-m", "--message", default="update")
    sub.add_parser("search-labels")
    p = sub.add_parser("push")
    p.add_argument("stage", choices=["cache", "labels", "train", "submit", "blend", "head"])
    p.add_argument("--slug")
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--llm-model", default=None)
    p.add_argument("--n-teachers", type=int, default=1)
    p.add_argument("--refine-from", default=None)
    p.add_argument("--config", default=None, help='JSON overrides for TRAIN, e.g. \'{"epochs": 8}\'')
    p.add_argument("--train", nargs="*", default=["kneemri-train-r0"])
    p.add_argument("--budget-hours", type=float, default=8.0)
    p.add_argument("--stack", action="store_true", help="submit stage: try the LightGBM stacker (rule-gated)")
    p.add_argument("--base", default=PUBLIC_BASE, help="public notebook to fork (blend stage)")
    p.add_argument("--weight", type=float, default=0.1, help="rank weight of our runs (blend stage)")
    p.add_argument("--img", type=int, default=None, help="cache stage: slice size (default VOLUME)")
    p.add_argument("--crop-mm", type=float, default=None, help="cache stage: field of view")
    p.add_argument("--cache", nargs="+", default=["kneemri-cache"],
                   help="train stage: cache kernel(s) to train on (several = shards of one cache)")
    p.add_argument("--shard", default=None, help="cache stage: i/n, cache every n-th study from i")
    p.add_argument("--run", default=None, help="head stage: stage-1 train kernel whose backbones to reuse")
    p.add_argument("--views", type=int, default=3, help="head stage: feature views per study (1 clean + augmented)")
    p.add_argument("--head-epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, default=2, help="head stage: heads per fold")
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
        version = a.version
        vfile = BUILD / f"{a.slug}.version"
        if not version and vfile.exists():
            version = vfile.read_text().strip()
        if not version:
            raise SystemExit(f"no known version for {a.slug}; pass -v (see the kernel's page)")
        print(f"submitting {a.slug} version {version}", flush=True)
        cmd += ["-v", version]
        run(cmd)


if __name__ == "__main__":
    main()
