# Stage 1 (CPU, internet on): decode every training study once into the .npz cache, and
# download the DICOM codec wheels that the offline submission notebook installs.
from kneemri.kaggle_env import find_competition_dir

DATA = find_competition_dir()
sh("pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg")
sh(f"mkdir -p {WORK}/wheels && pip download -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg "
   f"-d {WORK}/wheels")
sh(f"python {CODE}/scripts/cache_volumes.py --data {DATA} --split train --out {WORK}/cache "
   f"--img {CONFIG['img']} --crop-mm {CONFIG['crop_mm']} --workers {os.cpu_count() or 4}"
   + (" --recenter" if CONFIG.get("recenter") else "")
   + (f" --shard {CONFIG['shard']}" if CONFIG.get("shard") else ""))
n = len(list((WORK / "cache").glob("*.npz")))
print(f"cached {n} studies in {(time.time() - T0) / 60:.1f} min", flush=True)
json.dump({"img": CONFIG["img"], "crop_mm": CONFIG["crop_mm"], "recenter": bool(CONFIG.get("recenter"))},
          open(WORK / "cache" / "volume.json", "w"))
