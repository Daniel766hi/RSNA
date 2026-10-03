# RSNA Knee Abnormality Detection 2026

Our work on the Kaggle [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
challenge:

* **Task:** detect 12 knee-MRI findings.
* **Metric:** macro ROC AUC.
* **Data:** about 4.4k training studies, of which only about 58 carry labels. Every study has
  a free-text report; test studies have no report.
* **Deadlines:** entry and team merger by **2026-10-15**, final submission **2026-10-22**.

## Start here

1. **[`docs/research_review.md`](docs/research_review.md)**: a literature review (knee MRI deep
   learning, multiple-instance learning, report labelling, noisy labels, AUC optimisation,
   clinical co-occurrence priors) combined with public competition intelligence. It turns into
   a ranked list of levers (L1–L8) and a dated roadmap with pre-registered decision rules.
2. **[`docs/kaggle_runbook.md`](docs/kaggle_runbook.md)**: the exact Kaggle commands, compute
   budget and submission checklist.

## The approach in one paragraph

The public frontier (about 0.94 LB) uses a 2.5-D CoAtNet with a permutation-invariant
attention pool, trained on report-derived labels. We keep the 2.5-D backbone and target its
two measured weaknesses:

1. **Labels are systematically wrong where reports are silent.** This affects Synovitis,
   Effusion, Contusion and OA, where image models already beat every report-label source on
   gold. We therefore label reports with a present / absent / uncertain / *not mentioned*
   schema, then refine the silent cells with calibrated out-of-fold image predictions (L1).
2. **The pooling cannot tell where a slice lies.** This is what separates the medial from the
   lateral compartment. Our MIL head embeds each window's slot and its normalised slice
   position after right knees are mirrored to left. A window-context transformer, per-finding
   queries and a label-interaction layer then produce the 12 logits (L2, L3).

Everything is fused by rank mean, and it can be blended with the public CoAtNet family as a
decorrelated member (L6).

## Repository layout

```
docs/research_review.md      research synthesis, ranked strategy, roadmap, references
docs/kaggle_runbook.md       how to run each stage on Kaggle, budget, submission checklist
src/kneemri/
  series.py                  slot selection (plane x contrast) from *_series.csv
  volume.py                  DICOM -> sorted, canonical, laterality-normalised, physically
                             scaled uint8 stacks + slice positions; .npz cache
  data.py                    2.5-D triplet windows, study-consistent augmentation (no flips),
                             window/slot dropout, padding collate
  model.py                   KneeMIL: timm backbone -> slot+position embedding -> context
                             transformer -> 12 finding queries -> label interaction -> logits
  losses.py                  weighted soft BCE + in-batch pairwise AUC surrogate
  train.py                   report-hash-grouped CV, EMA, AMP, OOF export, gold-58 guard
  infer.py                   offline time-budgeted DICOM inference, one decode per study
  ensemble.py                rank-mean fusion
  metrics.py                 macro AUC, paired bootstrap for the small gold set
  io.py                      UID-safe CSV reading
  labels/teachers.py         teacher tables: load, gold-leak detection, gold agreement, fusion
  labels/llm_labeler.py      offline open-weight LLM labeler (vLLM / transformers)
  labels/refine.py           silence-aware label refinement with OOF image predictions
scripts/                     cache_volumes / label_reports_llm / train_cv / refine_labels / predict
tests/                       28 tests on synthetic DICOM series (orientation, laterality, scale,
                             labels, model, end-to-end train -> DICOM inference)
```

## Quick start (local / Kaggle)

```bash
pip install -e ".[dev]"      # plus pylibjpeg codecs for JPEG / JPEG 2000 DICOMs
pytest -q                    # 28 tests, CPU only, no competition data needed

DATA=/kaggle/input/rsna-knee-abnormality-detection
python scripts/cache_volumes.py --data $DATA --split train --out cache/train_384 --img 384
python scripts/label_reports_llm.py --data $DATA --model /kaggle/input/<llm> --out labels/llm.csv
python scripts/train_cv.py --data $DATA --cache cache/train_384 --labels labels/llm.csv --out outputs/r0
python scripts/refine_labels.py --data $DATA --labels labels/llm.csv --oof "outputs/r0/oof_fold*.csv" --out labels/r1.csv
python scripts/train_cv.py --data $DATA --cache cache/train_384 --labels labels/r1.csv --out outputs/r1
python scripts/predict.py --data $DATA --group "outputs/r1/fold*.pt" --out submission.csv
```

## Status and results

The pipeline now runs on the competition data on Kaggle (`kaggle/orchestrate.py`).

* **Teacher labels.** 5 public tables copied the gold labels and were dropped. The best clean
  table is `riadmohamed42/jev-knee-labels` `HYBRID_labels_scores.csv`, with 0.904 macro
  agreement with gold.
* **Cache.** All 4,407 training studies were decoded at 320 px in 77 min. A 352 px cache is
  being built.

| Run | Setup | Gold-58 macro (folds 0/2/4, 1/3) | Public LB |
|---|---|---|---:|
| r0 | ConvNeXt-T, 320 px, 5 folds × 10 epochs, teacher labels | 0.864 / 0.840 | 0.883 |
| r1 | as r0, labels refined with r0 OOF (L1) | 0.865 / 0.861 | — |
| r0+r1 | rank mean | — | 0.888 |
| r2 | ConvNeXt-S, 4 folds, labels refined with r0 OOF | 0.857 / 0.886 (29 + 29) | — |
| r0+r1+r2 | rank mean (gold-58 pooled 0.875); LightGBM stacker rejected by its rule (0.867) | — | **0.893** |
| r3 | ConvNeXt-T at 352 px (sharded cache), labels refined with r1 OOF | 0.869 / 0.864 | — |
| r0+r1+r2+r3 | rank mean, own pipeline only | — | **0.894** |
| blend | public 0.943 pipeline (`push blend`) + own r0-r3 at 10% rank weight | — | 0.941 (below the public pipeline alone: our members are too weak to add value yet) |
| LLM labels | Qwen2.5-14B-Instruct-AWQ (vLLM, 2xT4), present/absent/uncertain/not-mentioned; gold agreement 0.832 (Effusion 0.70, Synovitis 0.75), not leaky | — | — |
| r4 | ConvNeXt-T 320 px on LLM labels, silent cells refined with r1 OOF | 0.844 / 0.841: **rejected** (−0.02 vs r1 on both halves; the LLM teacher, 0.832 gold agreement, is too weak) | — |
| r5 | ConvNeXt-S, 352 px, 24 windows, 4 folds, labels refined with r3 OOF | 0.852 / 0.880 (r2: 0.857 / 0.886); kept as a member, but no gain for 11.3 h of GPU | — |
| r6 | ConvNeXt-T, 352 px, labels refined with the r1+r2+r3 OOF ensemble | running | — |
| r1 + stage 2 | r1 backbones frozen, MIL head retrained on all windows (`train_head.py`), same labels as r1 | 0.885 / 0.841 (weak 0.880 vs 0.881): **rejected**, gold flat and weak slightly worse; 4.1 h, mostly CPU-bound feature extraction | — |

Gold-58 has only 35 and 23 studies per half, so per-target values are noisy.

**For reference:** the best public notebook scores 0.943 (`yamadan96/rsna-knee-d4-public0946`).
`push blend` forks it and rank-blends our runs at 10%, which is how its parent reached 0.946.

Competition data, caches and checkpoints are never committed (Kaggle rules; see `.gitignore`).
