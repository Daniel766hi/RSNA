# Kaggle runbook

Run the stages in order. Each stage is a separate Kaggle notebook. **Training notebooks may use
the internet** (pip, timm weights); only the **submission notebook must run offline**.

## Automated route: `kaggle/orchestrate.py`

Every stage below also exists as a ready-made private script kernel (`kaggle/stages/`). The
orchestrator uploads the code, pushes the kernels with the right inputs chained, waits for
them, and submits. It needs `KAGGLE_API_TOKEN` in the environment (a token from
kaggle.com → Settings → API). The token must belong to an account that has **accepted the
competition rules** and is **phone-verified** (both are required for GPU and submissions).

```bash
python kaggle/orchestrate.py whoami
python kaggle/orchestrate.py upload-code -m "v1"
python kaggle/orchestrate.py search-labels                       # pick clean public label tables
python kaggle/orchestrate.py push cache            && python kaggle/orchestrate.py wait kneemri-cache
python kaggle/orchestrate.py push labels --datasets <owner>/<table> ...
python kaggle/orchestrate.py wait kneemri-labels                 # prints the leak / gold-agreement report
python kaggle/orchestrate.py push train            && python kaggle/orchestrate.py wait kneemri-train-r0
python kaggle/orchestrate.py push submit --train kneemri-train-r0 && python kaggle/orchestrate.py wait kneemri-submit
python kaggle/orchestrate.py submit kneemri-submit -m "r0 convnext-t 320"
python kaggle/orchestrate.py push train --slug kneemri-train-r1 --refine-from kneemri-train-r0
python kaggle/orchestrate.py push submit --train kneemri-train-r0 kneemri-train-r1   # then wait + submit
```

What each stage does:

* **Labels:** every attached table is leak-checked against the gold rows. Leaky tables are
  dropped, and the best clean table by gold agreement is used.
* **Train:** the stage splits folds across both T4s.
* **Submit:** the stage runs offline and installs the DICOM codec wheels that the cache stage
  downloaded.

The whole chain was simulated end-to-end outside Kaggle, on synthetic DICOM, with
`KNEEMRI_KAGGLE_ROOT` pointing at a fake `/kaggle`.

## 0. One-time setup (manual route)

1. **Accept the competition rules before 2026-10-15** (entry and team-merger deadline).
2. Upload this repository as a private Kaggle dataset, e.g. `kneemri-code`. In each notebook:
   `!pip install --no-deps -e /kaggle/input/kneemri-code`, or add
   `/kaggle/input/kneemri-code/src` to `sys.path`.
3. DICOM codecs. The series mix JPEG Lossless and JPEG 2000, and pydicom needs a decoder:
   * **Training notebooks:** `pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg`.
   * **Submission notebook:** download the same wheels once (`pip download ... -d wheels`),
     upload them as a dataset, and run `pip install --no-index --find-links /kaggle/input/<wheels> ...`.
   * Check: `scripts/cache_volumes.py` prints `FAILED` for any study it cannot decode.

## 1. Labels (about 1 h on GPU, or free if you reuse a clean public table)

* **Option A: LLM labels.** Attach an open-weight instruction model as a dataset (a 7–14B AWQ
  model fits on 2×T4 with vLLM), then run:

  ```bash
  python scripts/label_reports_llm.py --data $DATA --model /kaggle/input/<llm> --out labels/llm.csv
  ```

  The script prints per-target gold agreement and the leak check. Compare against the best
  public clean table (0.899 macro agreement).
* **Option B: public table(s).**
  1. Load them with `kneemri.labels.teachers.load_teacher`.
  2. **Run `detect_gold_leak` on each and discard any that are flagged.** Five public tables
     copy the gold labels.
  3. Fuse the clean ones with `fuse_teachers`.
  4. Where a table has no state information, `refine_labels.py` falls back to distance from
     0.5 as explicitness. Prefer Option A for L1.

## 2. Volume cache (CPU, about 2–3 h for 4.4k studies with 4 workers)

```bash
python scripts/cache_volumes.py --data $DATA --split train --out /kaggle/working/cache_384 --img 384 --crop-mm 140
```

* **Size:** at 384 px there are 58 uint8 slices per study, about 8.5 MB raw per study before
  compression. Save the cache as a dataset output so later sessions reuse it.
* **For the L4 A/B:** build a second, smaller cache on 2,000 studies with `--crop-mm 120 --recenter`.

## 3. Round-0 CV

Training cost on a T4 in fp16, 384 px, 16 random windows per study per step, using measured
throughputs from the public ledger:

| Backbone | img/s fwd+bwd | min/epoch (4.35k studies) | 5 folds × 10 epochs on 2×T4 |
|---|---:|---:|---:|
| ConvNeXt-T | ~59–75 | ~16–20 | ~7–8 h |
| EfficientNetV2-S | ~51–68 | ~17–23 | ~7–9 h |
| ConvNeXt-S (grad-ckpt) | ~36 | ~32 | ~13 h (two sessions) |
| CoAtNet-2 (grad-ckpt) | ~13 | ~90 | too slow for 5-fold CV here |

Recommendations:

* Run round 0 with **ConvNeXt-T or EfficientNetV2-S at 384**, 3–5 folds.
* Run two folds in parallel, one per GPU:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train_cv.py ... --folds 0 2 4 --out outputs/r0 &
  CUDA_VISIBLE_DEVICES=1 python scripts/train_cv.py ... --folds 1 3   --out outputs/r0 &
  ```

  Every fold writes its own `oof_fold<k>.csv`, so parallel processes never collide.
  `refine_labels.py --oof "outputs/r0/oof_fold*.csv"` concatenates them.
* Default hyperparameters:

  ```bash
  python scripts/train_cv.py --data $DATA --cache cache_384 --labels labels/llm.csv \
    --backbone convnext_tiny.fb_in22k_ft_in1k_384 --epochs 10 --batch-size 8 --out outputs/r0
  ```

* **Ablations**, each on 1 fold and a 2,000-study subset (`--limit 2000 --folds 0`):
  * `--no-position` (L2 off);
  * `--label-layers 0` (L3 off);
  * `--auc-weight 0` (L5 off).

  Keep a component only if the weak-label hold-out macro is not worse and gold-58 does not
  drop by more than 0.01. Compare two `oof.csv` files with `kneemri.metrics.paired_bootstrap`.

## 4. Round 1: label refinement (L1)

```bash
python scripts/refine_labels.py --data $DATA --labels labels/llm.csv --oof "outputs/r0/oof_fold*.csv" --out labels/r1.csv
python scripts/train_cv.py ... --labels labels/r1.csv --out outputs/r1
```

* The refined labels change mostly on the silent cells. The script prints the mean change per
  target. Expect the biggest changes on Synovitis, OA and Contusion.
* **Decision rule:** keep r1 if weak-label OOF improves and the gold-58 paired-bootstrap delta
  has its lower 90% bound above −0.01.

## 5. Submission notebook (offline, ≤ 9 h, 2×T4)

```bash
python scripts/predict.py --data /kaggle/input/rsna-knee-abnormality-detection \
  --group "/kaggle/input/kneemri-r1/fold*.pt" --budget-hours 8 --workers 4 --out submission.csv
```

* **Budget:** decode takes about 1 s/study on CPU, overlapped with the GPU through the
  DataLoader workers. The model takes about 0.1–0.3 s/study per fold for ConvNeXt-T on a T4.
  Five folds × 1,300 studies ≈ 40–70 min. Measure it on the first submission.
* **L6, blending with the public CoAtNet family:**
  1. Run the public family in the same notebook first, writing its predictions to a CSV.
  2. Pass that CSV as `--blend-with coatnet.csv --blend-weight 0.5`. Use an equal share; do
     not tune it on the LB.
* **Safety:** undecodable studies and studies past the time budget get 0.5 rather than
  crashing the run. Check the notebook log for `decode failed`.

## 6. Final selection (2026-10-20 to 2026-10-22)

Pick two submissions:

* (a) the best public-LB score;
* (b) the most principled one: equal weights, no LB-probed choices.

The public LB covers only part of the ~1,300 test studies, and prevalence differs between the
public and private sets. Hedging against shake-up is worth more than the last +0.001 on the
public LB.
