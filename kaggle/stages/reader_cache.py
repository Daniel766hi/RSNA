# Reader stage 1 (CPU): decode training series with the public 2.5-D ConvNeXt reader's own
# preprocessing (goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache-2.0), unchanged, so the
# retrained reader sees exactly what its inference code feeds it. Volumes are stored compressed
# (one .npz per series) and sharded by study to stay under the 20 GB output limit.
import importlib.util

import numpy as np
import pandas as pd
from multiprocessing import Pool

from kneemri.kaggle_env import find_competition_dir

DATA = find_competition_dir()
sh("pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg", check=False)
SRC = next(p.parent for p in sorted(INPUT.rglob("preprocess.py")) if (p.parent / "knee.py").exists())
spec = importlib.util.spec_from_file_location("reader_preprocess", SRC / "preprocess.py")
pre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre)
print("reader source:", SRC, "| SIZE", pre.SIZE, "MM_PER_PX", pre.MM_PER_PX, "MAX_SLICES", pre.MAX_SLICES, flush=True)

OUT = WORK / "reader_cache"
OUT.mkdir(exist_ok=True)
se = pd.read_csv(DATA / "train_series.csv")
studies = sorted(se.StudyInstanceUID.unique())
if CONFIG.get("shard"):
    i, n = map(int, CONFIG["shard"].split("/"))
    keep = set(studies[i::n])
    se = se[se.StudyInstanceUID.isin(keep)]
if CONFIG.get("limit"):
    se = se.sample(int(CONFIG["limit"]), random_state=0)


def job(row):
    study, series, plane = row
    out = OUT / f"{series}.npz"
    info = {"study": study, "series": series}
    t = time.time()
    try:
        vol, extra = pre.load_series(str(DATA / "train_series" / study / series), plane)
        np.savez_compressed(out, v=vol)
        info.update(extra, raw_mb=vol.nbytes / 2**20, npz_mb=out.stat().st_size / 2**20)
    except Exception as e:  # noqa: BLE001 - recorded per series, the reader skips it
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    info["sec"] = time.time() - t
    return info


rows = []
tasks = list(zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane))
with Pool(os.cpu_count() or 4) as p:
    for k, r in enumerate(p.imap_unordered(job, tasks, chunksize=2)):
        rows.append(r)
        if k % 500 == 0:
            print(f"{k}/{len(tasks)} {(time.time() - T0) / 60:.1f} min", flush=True)
meta = pd.DataFrame(rows)
meta.to_csv(OUT / "meta.csv", index=False)
ok = meta[meta["error"].isna()] if "error" in meta.columns else meta
print(f"[reader_cache] {len(ok)}/{len(meta)} series ok in {(time.time() - T0) / 60:.1f} min; "
      f"raw {ok.raw_mb.sum() / 1024:.2f} GB, npz {ok.npz_mb.sum() / 1024:.2f} GB; "
      f"mean slices {ok.n.mean():.1f}; mean {meta.sec.mean():.2f} s/series/proc", flush=True)
if CONFIG.get("limit"):  # size probe: project the full cache and drop the files
    f = len(pd.read_csv(DATA / "train_series.csv")) / max(len(meta), 1)
    print(f"[reader_cache] projected full cache: npz {ok.npz_mb.sum() * f / 1024:.1f} GB, "
          f"{meta.sec.sum() * f / (os.cpu_count() or 4) / 3600:.1f} h on {os.cpu_count()} procs", flush=True)
    for p in OUT.glob("*.npz"):
        p.unlink()
