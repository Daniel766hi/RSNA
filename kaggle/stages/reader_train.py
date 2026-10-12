# Reader stage 3 (2xT4, internet on for timm weights): train (or, with CONFIG["eval"], score the
# published checkpoints of) the 2.5-D ConvNeXt reader, one fold per GPU in parallel.
import torch

from kneemri.kaggle_env import find_competition_dir

DATA = find_competition_dir()
SRC = next(p.parent for p in sorted(INPUT.rglob("knee.py")) if (p.parent / "train.py").exists())
caches = sorted({str(p.parent) for p in INPUT.rglob("meta.csv")
                 if p.parent.name == "reader_cache" and any(f"/{c}/" in str(p) for c in CONFIG["caches"])})
labels = next(p for p in INPUT.rglob(f"labels_{CONFIG['labels']}.csv") if f"/{CONFIG['labels_kernel']}/" in str(p))
print("reader src:", SRC, "| caches:", caches, "| labels:", labels, flush=True)
if len(caches) != len(CONFIG["caches"]):
    raise SystemExit(f"expected {len(CONFIG['caches'])} cache shards, found {caches}")

n_gpu = max(1, torch.cuda.device_count())
folds = CONFIG["folds"]
out = WORK / "run"
base = (f"python {CODE}/scripts/reader_train.py --src {SRC} --data {DATA} --cache {','.join(caches)} "
        f"--labels {labels} --out {out} --workers {max(1, (os.cpu_count() or 4) // n_gpu)} "
        f"--micro {CONFIG.get('micro', 2)} --epochs {CONFIG.get('epochs', 14)} --seed {CONFIG.get('seed', 0)}"
        + (f" --limit {CONFIG['limit']}" if CONFIG.get("limit") else ""))
procs = []
for g in range(n_gpu):
    mine = [k for i, k in enumerate(folds) if i % n_gpu == g]
    if not mine:
        continue
    cmds = []
    for k in mine:
        ck = f" --eval-ckpt {SRC}/cnxt_v0_fold{k}.pt" if CONFIG.get("eval") else ""
        cmds.append(f"{base} --fold {k}{ck} > {WORK}/log_g{g}_f{k}.txt 2>&1")
    cmd = f"CUDA_VISIBLE_DEVICES={g} bash -c '{' && '.join(cmds)}'"
    print("$", cmd, flush=True)
    procs.append((g, subprocess.Popen(cmd, shell=True)))
rc = 0
for g, p in procs:
    rc |= p.wait()
for f in sorted(WORK.glob("log_g*_f*.txt")):
    lines = f.read_text().splitlines()
    print(f"---- {f.name} (last 40 lines)", flush=True)
    print("\n".join(lines[-40:]), flush=True)
for f in sorted(out.glob("metrics_fold*.json")):
    print(f"[reader_train] {f.name}: {f.read_text()}", flush=True)
print(f"[reader_train] done in {(time.time() - T0) / 3600:.2f} h", flush=True)
if rc:
    raise SystemExit(f"a fold failed (rc {rc})")
