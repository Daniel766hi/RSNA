# ---- appended by kaggle/orchestrate.py (push blend): fuse our KneeMIL runs into the public
# pipeline's submission.csv by weighted rank mean (lever L6, our model as the decorrelated
# member). Any failure here leaves the public submission untouched.
import gc as _km_gc, os as _km_os, subprocess as _km_sp, sys as _km_sys, time as _km_time
from pathlib import Path as _KmPath

import pandas as _km_pd

import json as _km_json

_KM_CFG = _km_json.loads(r'''__CONFIG__''')
_KM_T0 = globals().get("T0", None)
_km_sub_path = _KmPath("/kaggle/working/submission.csv")
try:
    if not _km_sub_path.exists():
        raise RuntimeError("public pipeline wrote no submission.csv")
    try:
        import torch as _km_torch
        _km_gc.collect()
        _km_torch.cuda.empty_cache()
    except Exception:
        pass
    _km_in = _KmPath("/kaggle/input")
    _km_code = next(p.parent.parent.parent for p in sorted(_km_in.rglob("kneemri/__init__.py"))
                    if p.parent.parent.name == "src")
    _km_sys.path.insert(0, str(_km_code / "src"))
    from kneemri.kaggle_env import find_competition_dir as _km_comp

    _km_data = _km_comp()
    _km_wheels = sorted(_km_in.rglob("pylibjpeg*.whl"))
    if _km_wheels:
        _km_sp.run(f"pip install -q --no-index --find-links {_km_wheels[0].parent} "
                   "pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg", shell=True)
    _km_groups = sorted({str(p.parent) for p in _km_in.rglob("fold*.pt") if "kneemri-train" in str(p)})
    if not _km_groups:
        raise RuntimeError("no kneemri checkpoints attached")
    _km_elapsed = (_km_time.time() - _KM_T0) / 3600 if isinstance(_KM_T0, (int, float)) else 0.0
    _km_budget = max(0.25, min(_KM_CFG["max_hours"], _KM_CFG["total_hours"] - _km_elapsed))
    print(f"[kneemri] groups={_km_groups} elapsed={_km_elapsed:.2f} h budget={_km_budget:.2f} h", flush=True)
    def _km_spec(g):  # runs trained on a non-default cache record their volume recipe
        v = _KmPath(g) / "volume.json"
        return "@{img}:{crop_mm}".format(**_km_json.loads(v.read_text())) if v.exists() else ""

    _km_out = _KmPath("/kaggle/working/kneemri_pred.csv")
    _km_cmd = ([_km_sys.executable, str(_km_code / "scripts" / "predict.py"), "--data", str(_km_data)]
               + sum((["--group", f"{g}/fold*.pt{_km_spec(g)}"] for g in _km_groups), [])
               + ["--img", str(_KM_CFG["img"]), "--crop-mm", str(_KM_CFG["crop_mm"]),
                  "--budget-hours", f"{_km_budget:.3f}", "--workers", str(_km_os.cpu_count() or 4),
                  "--out", str(_km_out), "--covered-out", str(_km_out.with_name("kneemri_covered.csv"))])
    _km_env = dict(_km_os.environ, PYTHONPATH=str(_km_code / "src"))
    _km_r = _km_sp.run(_km_cmd, env=_km_env)
    if _km_r.returncode != 0 or not _km_out.exists():
        raise RuntimeError(f"predict.py failed ({_km_r.returncode})")

    _km_pub = _km_pd.read_csv(_km_sub_path, dtype={"StudyInstanceUID": str})
    _km_ours = _km_pd.read_csv(_km_out, dtype={"StudyInstanceUID": str})
    _km_cols = [c for c in _km_pub.columns if c != "StudyInstanceUID"]
    _km_ours = _km_pub[["StudyInstanceUID"]].merge(_km_ours, on="StudyInstanceUID", how="left")
    # studies our run skipped (budget / decode failure) keep the public prediction only
    _km_cov = set(_km_pd.read_csv(_km_out.with_name("kneemri_covered.csv"), dtype={"StudyInstanceUID": str})
                  ["StudyInstanceUID"])
    _km_have = _km_pub["StudyInstanceUID"].isin(_km_cov) & ~_km_ours[_km_cols].isna().any(axis=1)
    _km_w = _KM_CFG["weight"] * _km_have.to_numpy(float)[:, None]
    _km_blend = _km_pub.copy()
    _km_blend[_km_cols] = ((1 - _km_w) * _km_pub[_km_cols].rank(pct=True).to_numpy()
                           + _km_w * _km_ours[_km_cols].rank(pct=True).fillna(0.5).to_numpy())
    _km_tmp = _km_sub_path.with_suffix(".csv.tmp")
    _km_blend.to_csv(_km_tmp, index=False)
    _km_os.replace(_km_tmp, _km_sub_path)
    print(f"[kneemri] blended weight={_KM_CFG['weight']} into {_km_sub_path} "
          f"({int(_km_have.sum())}/{len(_km_have)} studies with our prediction)", flush=True)
except Exception as _km_exc:  # noqa: BLE001 - never lose the public submission
    print(f"[kneemri] blend skipped, public submission kept: {_km_exc!r}", flush=True)
