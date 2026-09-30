# RSNA Knee Abnormality Detection 2026: research review and winning strategy

*Written 2026-09-30, three weeks before the final submission deadline (2026-10-22). The
entry and team-merger deadline is **2026-10-15**, so accept the rules before then.*

This document does three things:

1. It pins down what the competition actually rewards (§1–2).
2. It reviews the literature and the public competition intelligence, theme by theme. Each
   theme ends with what it means for this competition (§3–4).
3. It turns that into a ranked plan with decision rules (§5–6). The code in `src/kneemri/`
   implements the levers the plan ranks highest.

---

## 1. The competition in one page

| Item | Fact |
|---|---|
| Task | Per-study probability of 12 findings: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's (cyst), Contusion, Fracture |
| Metric | Macro-averaged ROC AUC over the 12 targets. Only the ranking within each target matters, so calibration and between-target scale are irrelevant. |
| Train data | 4,407 studies, about 24k series (median 30 slices per series). `train.csv` has a free-text `Report` (9+ languages) for every study, but **only ~58 studies (1.3%) carry the 12 labels** |
| Series metadata | `train_series.csv` gives `Anatomical_Plane` (Sag/Cor/Ax), `Fluid_Sensitive`, `Fat_Suppression` per series. PD fat-sat is the most common series type. |
| Test data | About 1,300 studies. **No report at test time**: the text is training-time-only (privileged) information. |
| Runtime | Kaggle notebook, internet off, 9 h on CPU or GPU (2×T4 in practice; the P100 does not work with the current torch build) |
| Prevalence | The host warns that prevalence differs across train, public and private sets. AUC per target is invariant to prevalence, so what matters is ranking robustness. |
| Efficiency track | A separate prize ($18k total) that trades score against runtime. On the public efficiency board, the #1 most efficient team also scores 0.954 on the LB. |
| Leaderboard (mid-Sep) | Top about 0.955, best public notebooks 0.939–0.941, about 2,400 teams (a public repo reports 0.935 as rank about 206) |

**Two structural facts drive every decision below:**

* **This is weakly supervised learning.** The imaging model learns from labels mined from
  reports. The 58 gold studies (macro-AUC standard error about 0.02–0.03) can catch large
  regressions but cannot tune ±0.01 effects.
* **The gold and test labels appear to be image-based.** Image models beat every
  report-derived label source on Effusion, Synovitis, Contusion and Medial OA when both are
  scored against gold (public ledger, A02). Report labels therefore carry *systematic* noise
  on the findings radiologists often leave out of reports. This is the single most
  exploitable fact in the competition.

## 2. Where the current public frontier loses points

A public study of the 0.94 ensemble (TranBaDat2607, `docs/report.md` in their repo) measured
the best single public CoAtNet per finding on gold-58:

| Finding | Gold-58 AUC (best single model) | Report-label agreement with gold (best public table) |
|---|---:|---:|
| Synovitis | **0.80** | 0.79 |
| PF OA | **0.84** | 0.90 |
| Lateral OA | **0.85** | 0.83 |
| MCL | 0.89 | — |
| Lateral Meniscus | 0.90 | — |
| Fracture | 0.91 | 0.87 |
| Contusion | 0.92 | 0.86 |
| all others | ≥ 0.95 | — |

What else that study (and a second public repo) measured:

* The strongest representation is a **2.5-D attention-MIL** model: a CoAtNet-RMLP-2 at
  336–384 px on slice triplets, 44–64 slices over 5 plane×contrast slots, pooled by a
  per-finding attention head. One checkpoint scores 0.91–0.92 on gold-58 and about 0.914 on
  the LB.
* **The diversity that pays is pipeline and label diversity, not backbone diversity.**
  * Swapping the backbone with the same labels and views gained +0.001 on the LB.
  * Checkpoints from the same pipeline correlate at Spearman 0.91–0.94.
  * A CoAtNet from a different author, with different labels and preprocessing, added
    +0.007 on gold-58. The weak but different DINO/RadImageNet chain lifted the LB from
    about 0.92 to 0.94.
* **The attention pool is permutation-invariant.** Reversing the window order changes
  nothing (max diff 2.6e-5). **The frontier models therefore have no notion of *where* in
  the knee a slice lies.** For sagittal slices, that position is exactly what separates the
  medial from the lateral compartment. This is our reading of their result, and it is
  consistent with Lateral OA and Lateral Meniscus being among the weakest findings. See
  lever L2.
* Retraining only the head on a better teacher moved **Synovitis by +0.067**, but little
  else (the "teacher" is the report-label table used as training targets). The Raptor
  CoAtNets were trained on labels that disagree most with the best teachers on Synovitis
  (Spearman 0.52), Fracture (0.55–0.61), Contusion (0.75), MCL (0.78) and Baker's (0.79).
* **Five public label tables copy the 58 gold labels verbatim.** Any gold-58 validation that
  uses them is silently leaked.

The remaining 0.94 → 0.955+ gap is therefore not "more ensemble members". It is a better
core model on a handful of findings. The efficiency board shows the same thing: 0.954 in a
runtime-efficient submission.

---

## 3. Literature review, theme by theme

### 3.1 Deep learning on knee MRI

* **MRNet** ([Bien et al., PLOS Med 2018](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699)).
  * Method: one 2-D AlexNet per plane, max-pooled over slices, with a logistic regression
    across the three planes.
  * Results: AUC 0.965 (ACL) and 0.847 (meniscus) on 1,370 exams.
  * Lessons: slice-wise 2-D features plus slice pooling are a strong baseline; **planes
    contribute differently per finding** (ACL: sagittal; MCL: coronal; meniscus: sagittal
    and coronal).
* **MRPyrNet** ([Dunnhofer et al., MIDL 2021](https://proceedings.mlr.press/v143/dunnhofer21a.html); extended in [CMIG 2022](https://pubmed.ncbi.nlm.nih.gov/36446308/)).
  * Knee lesions are small and sit in predictable regions.
  * Feature-pyramid detail pooling improves ACL and meniscus detection over MRNet and ELNet.
  * Lesson: **effective resolution on the joint matters.** Centred crops that discard native
    detail cost the resolution-limited findings (menisci, cartilage/OA, MCL).
* **Astuto et al., Radiology: AI 2021** ([link](https://pubs.rsna.org/doi/full/10.1148/ryai.2021200165)).
  * Method: hierarchical 3-D CNNs that first localise anatomic compartments (cartilage,
    bone marrow, meniscus, ACL), then detect and grade per compartment.
  * Results: AUC 0.83–0.93; bone-marrow oedema was the hardest (70% sensitivity).
  * Lesson: **compartment-aware (medial/lateral/patellofemoral) modelling** is how the
    literature handles medial-vs-lateral findings. Contusion (bone-marrow oedema) is
    intrinsically hard.
* **fastMRI+** ([Zhao et al., Sci Data 2022](https://www.nature.com/articles/s41597-022-01255-z)).
  * Content: 16,154 bounding-box annotations over 22 knee pathology categories (meniscus
    tear, ACL, bone-marrow oedema, effusion, cartilage and more) on sagittal PD-FS slices of
    the fastMRI knee set.
  * Lesson: this is the only large public source of *image-level* knee-pathology labels, so
    it is a candidate for pre-training and auxiliary supervision. Check the fastMRI data
    agreement against the competition's "reasonably accessible, minimal cost" external-data
    rule, and ask on the forum before relying on it.
* **Other related work.**
  * Pedoia et al. (Radiology 2019) and Namiri et al. (Radiology: AI 2020) use 3-D CNN
    severity staging for meniscus and ACL on OAI data.
  * Liu et al. (Radiology 2018/2019) built fully automated cartilage-lesion and ACL-tear
    detection. Their lesson is a segmentation-then-classification cascade.
  * ELNet (Tsai et al., MIDL 2020) is a small knee-specific CNN that is competitive with
    ImageNet backbones. It supports the efficiency track.

### 3.2 Aggregating many slices and series into one study prediction

* **Attention-based MIL** ([Ilse et al., ICML 2018](https://arxiv.org/abs/1802.04712)): gated
  attention pooling over instances. This is the pooling used by the frontier CoAtNets.
* **TransMIL** ([Shao et al., NeurIPS 2021](https://arxiv.org/abs/2106.00908)): a
  transformer over instances with positional encoding to model instance *correlation and
  position*. Plain attention-MIL discards both.
* **Query2Label** ([Liu et al., 2021](https://arxiv.org/abs/2107.10834)) and **ML-Decoder**
  ([Ridnik et al., WACV 2023](https://arxiv.org/abs/2111.12933)): one learnable query per
  label, cross-attending to spatial or instance tokens. This beats global pooling for
  multi-label problems in which each label lives in a different region. Here, "region"
  means plane, sequence and slice position.
* **Kaggle RSNA precedent.**
  * The 2022 cervical spine, 2023 abdominal trauma and 2024 lumbar spine challenges were
    won by **2.5-D CNNs** (neighbouring slices as channels) plus a sequence or attention
    aggregator.
  * Top lumbar-spine solutions used **localise, crop at high resolution, then classify**
    ([1st-place write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)).
* **Implication.** Keep the 2.5-D backbone and per-finding queries, and add what they
  discard:
  * slot (plane × contrast) embeddings;
  * a **normalised slice-position embedding**, which after laterality normalisation encodes
    medial↔lateral on sagittal and anterior↔posterior on coronal;
  * a light transformer over window tokens.

### 3.3 Turning reports into labels

* **CheXpert labeler** ([Irvin et al., AAAI 2019](https://arxiv.org/abs/1901.07031)) and
  **CheXbert** ([Smit et al., EMNLP 2020](https://arxiv.org/abs/2004.09167)).
  * Rule-based, then BERT-based, extraction with three states: positive, negative,
    uncertain. Silence is a separate "blank" state.
  * Lesson: *uncertain* and *not mentioned* must not be collapsed into negative.
* **Open-weight LLM labelling.**
  * Vicuna-13B matched CheXpert/CheXbert on 9/11 findings (median AUC 0.84) running fully
    locally ([Mukherjee et al., Radiology 2023](https://pubs.rsna.org/doi/full/10.1148/radiol.231147)).
  * Open-source LLMs extract radiology-report information reliably
    ([Le Guellec et al., Radiology: AI 2024](https://pubs.rsna.org/doi/full/10.1148/ryai.230364);
    [Radiology 2024 comparison](https://pubs.rsna.org/doi/10.1148/radiol.241139)).
  * **In this competition:** GPT-class labels reach 0.855–0.87 macro agreement with gold,
    and the best hybrid public table reaches 0.899. The gap is concentrated in Synovitis
    (0.62–0.79).
* **Implication.**
  * Label with a three-state-plus-silence schema.
  * Treat *silence* as "unknown", with a low sample weight and a prior, not as negative.
  * Treat the report as one noisy teacher among several (lever L1).

### 3.4 Learning with noisy and weak labels

* **Label-noise techniques.**
  * Bootstrapping loss ([Reed et al., 2015](https://arxiv.org/abs/1412.6596)) mixes the
    target with the model's own prediction.
  * Generalised cross-entropy ([Zhang & Sabuncu, NeurIPS 2018](https://arxiv.org/abs/1805.07836))
    is robust to noisy labels.
  * Co-teaching ([Han et al., NeurIPS 2018](https://arxiv.org/abs/1804.06872)) and
    DivideMix ([Li et al., ICLR 2020](https://arxiv.org/abs/2002.07394)) separate clean
    from noisy samples.
  * Early-learning regularisation ([Liu et al., NeurIPS 2020](https://arxiv.org/abs/2007.00151))
    exploits the fact that networks fit clean labels before memorising noise.
* **Surveys.**
  * [Song et al., IEEE TNNLS 2022](https://arxiv.org/abs/2007.08199) survey learning from
    noisy labels.
  * [Karimi et al., MedIA 2020](https://arxiv.org/abs/1912.02911) cover noisy labels in
    medical imaging specifically.
* **Self-training and distillation.**
  * Noisy Student ([Xie et al., CVPR 2020](https://arxiv.org/abs/1911.04252)) iterates
    teacher → pseudo-label → larger noised student.
  * Knowledge distillation ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) trains
    a student on soft teacher outputs.
* **Implication (lever L1: silence-aware label refinement).** The report noise here is
  *structured*: it is concentrated where the report is silent or vague, and the image shows
  the finding anyway.
  1. Train round-0 models on report labels.
  2. Build refined targets:
     `y* = w·y_report + (1−w)·p_image_OOF`
     * `w` is high when the report explicitly asserts or negates the finding.
     * `w` is low when the report is silent.
     * `p_image_OOF` is an out-of-fold ensemble prediction, so a study never teaches itself.
  3. Retrain round 1.

  This is bootstrapping and Noisy-Student at study level, targeted at exactly the findings
  where A02 showed report labels fail and images succeed: Synovitis, Effusion, Contusion
  and OA.

### 3.5 Using the report as privileged information

* **Learning Using Privileged Information** ([Vapnik & Vashist, 2009](https://doi.org/10.1016/j.neunet.2009.06.042)):
  information that is available only at training time can shape the learner. Distillation
  from a text or multimodal teacher is the modern form.
* **Image–report contrastive learning.** ConVIRT ([Zhang et al., 2020](https://arxiv.org/abs/2010.00747)),
  GLoRIA ([Huang et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html))
  and BioViL ([Boecking et al., ECCV 2022](https://arxiv.org/abs/2204.09817)) learn image
  features aligned to report text, and give label-efficient downstream classifiers.
* **Implication.** Beyond scalar labels, the report can supply:
  * graded certainty (e.g. "partial tear" versus "complete rupture");
  * severity for OA grading, used as an ordinal auxiliary target;
  * location (posterior horn, femoral condyle).

  An auxiliary head that predicts these at training time costs nothing at inference. It is
  lower priority than L1 because the head-only teacher swap moved only one finding.

### 3.6 Optimising and ensembling for AUC

* **Deep AUC maximisation** ([Yuan et al., ICCV 2021, LibAUC](https://arxiv.org/abs/2012.03173)):
  a margin-based AUC surrogate improves AUC over cross-entropy, especially on imbalanced
  medical labels. Add a small in-batch pairwise ranking term to BCE, in the spirit of a
  composite loss.
* **Rank averaging.** The metric is rank-based per target, so rank-mean (or mean of logits)
  across models is safer than probability averaging when members differ in calibration.
  This matters especially under the host's warning about prevalence shift.
* **Implication.** Use rank-mean fusion, fixed equal member weights, and no weights probed
  on the LB. The public ledger shows that tuned weights add ≤ 0.002 and risk private-LB
  shake-up.

### 3.7 Clinical co-occurrence priors (why a label-interaction layer helps)

* **Pivot-shift contusions and ACL.** Bone-contusion patterns are the "footprint of the
  mechanism of injury" ([Sanders et al., RadioGraphics 2000](https://pubs.rsna.org/doi/abs/10.1148/radiographics.20.suppl_1.g00oc19s135)).
  Lateral femoral condyle and posterolateral tibial-plateau bruises are up to ~97–100%
  specific for ACL rupture ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445054/)).
  So *Contusion ↔ ACL ↔ Lateral Meniscus* co-vary.
* **Synovitis on non-contrast MRI.** Semi-quantitative scoring (MOAKS; Hunter et al.,
  Osteoarthritis Cartilage 2011) scores synovitis as *Hoffa-synovitis* (signal change in
  the infrapatellar fat pad) and *effusion-synovitis*.
  * The information lives in anterior mid-sagittal slices and on fluid-sensitive sequences.
  * It is tightly coupled to Effusion, and to Baker's cyst, which communicates with the
    joint and co-occurs with effusion and degenerative disease.
* **OA compartments.** OA co-varies with degenerative meniscal tears in the same
  compartment.
* **Implication.** A small self-attention layer across the 12 per-finding tokens lets
  confident findings inform weak ones before the final logits. Examples: effusion informs
  synovitis; lateral contusion informs ACL.
  * Crops must keep Hoffa's fat pad and the suprapatellar recess. Aggressive
    tibiofemoral-only crops would hurt Synovitis and Effusion.

### 3.8 Localisation and cropping

* TotalSegmentator MRI ([Akinci D'Antonoli et al., Radiology 2025](https://pubs.rsna.org/doi/full/10.1148/radiol.241613))
  gives sequence-independent segmentation of 80 structures. Knee-specific cartilage and
  bone segmenters exist (OAI-ZIB, SKI10).
* A single joint centred by the coil does not need a detector. However, a cheap
  foreground-centroid re-centring followed by a *tighter physical crop* (about 110–120 mm
  instead of 140 mm) at the same pixel grid raises effective resolution on menisci and
  cartilage. Candidate findings are menisci, OA and MCL. This is hypothesis H2 in the public
  ledger, still untested there.

### 3.9 Efficiency track

* Per the public ledger, decoding (0.6–1.8 s/study on CPU) costs as much as the model.
  * Decode each study **once** and share it across all ensemble members.
  * Parallelise decoding across CPU workers while the GPU runs.
* A single student **distilled from the ensemble** (Hinton 2015) at a smaller backbone
  (ConvNeXt-T or EfficientNetV2-S at 3–5× CoAtNet-2 throughput on a T4) is the standard
  route to a top efficiency score. The #1 efficiency team at 0.954 suggests it is feasible.

---

## 4. What the evidence says *not* to do

* More checkpoints of the same pipeline, or more backbones on the same labels and views:
  each adds about +0.001.
* Per-target blend weights probed on the public LB or fitted on gold-58: shake-up risk, and
  within noise.
* Validating on public label tables that leak gold labels. Always run
  `kneemri.labels.teachers.detect_gold_leak` first.
* Horizontal or vertical flips as augmentation: they swap medial/lateral or femur/tibia.
  Mirroring right knees to a canonical left knee is *normalisation*, not augmentation.
* Treating report silence as a negative.

---

## 5. Ranked strategy

Expected gain is a judgement from the evidence above, not a measurement.

| # | Lever | Findings it targets | Evidence | Expected gain | Cost | Implemented in |
|---|---|---|---|---|---|---|
| **L1** | Silence-aware label refinement: report teacher fusion plus OOF image pseudo-labels on silent findings, over 2 rounds | Synovitis, Effusion, Contusion, OA, Fracture | A02 (image > report on these); Reed 2015; Noisy Student; head-only teacher swap +0.067 on Synovitis | +0.005–0.015 | 1 extra training round | `labels/teachers.py`, `labels/refine.py`, `scripts/refine_labels.py` |
| **L2** | Position-aware MIL: normalised slice-position and slot embeddings, window-context transformer, laterality normalisation | Lateral/Medial Meniscus, Lateral OA, MCL | Frontier pool is permutation-invariant; TransMIL; Astuto (compartment-aware) | +0.005–0.01 | none at inference | `model.py`, `volume.py` |
| **L3** | Label-interaction layer over per-finding queries (Query2Label / ML-Decoder style) | Synovitis (via Effusion), ACL/Contusion | §3.7 clinical priors; Query2Label | +0.002–0.005 | negligible | `model.py` |
| **L4** | Joint-centred tighter crop (≈ 120 mm) at ≥ 384 px | Menisci, OA, MCL | MRPyrNet; ledger H2 | +0.003–0.008 | same | `volume.py` (`crop_mm`, `recenter`) |
| **L5** | BCE plus in-batch pairwise AUC term; EMA; rank fusion | all | LibAUC; ledger | +0.001–0.003 | none | `losses.py`, `ensemble.py` |
| **L6** | Add the new pipeline as one *decorrelated* member next to the public CoAtNet family (equal weight, rank mean) | all | ledger: pipeline diversity is what adds | +0.003–0.01 over the base | about 1–1.5 h runtime | `scripts/predict.py` (`--blend-with`) |
| L7 | Report-derived auxiliary targets (severity, location) | OA, menisci | ConVIRT/BioViL, LUPI | +0–0.005 | medium | LLM schema already returns certainty; head is future work |
| L8 | Distilled single student for the efficiency track | — | Hinton 2015; efficiency board | efficiency prize | 1 training run | `refine_labels.py --w-explicit 0 --w-silent 0` on the ensemble OOF gives pure distillation targets for `train_cv.py` |

## 6. Roadmap to 2026-10-22, with decision rules

**Validation protocol (pre-registered, for every experiment):**

1. **Primary: a 5-fold OOF macro AUC against the *refined* weak labels** on all 4,349
   non-gold studies. Folds are grouped by report-text hash.
2. **Guard: gold-58 OOF macro AUC** with a paired bootstrap 90% CI
   (`kneemri.metrics.paired_bootstrap`). A change ships only if gold-58 does not drop by
   more than 0.01 *and* the primary metric improves.
3. **Arbiter: one public-LB submission** per shipped change. It is never used to tune
   weights.

| Dates | Step | Go / no-go |
|---|---|---|
| Oct 1–2 | Build the teacher table. Run `detect_gold_leak` on every public table and keep only clean ones. Optionally LLM-label with the open-weight labeler on Kaggle. Cache volumes with `scripts/cache_volumes.py` (decode once, about 2–3 h CPU). | Teacher-fusion gold agreement ≥ the best single clean table |
| Oct 2–5 | **Round 0:** `scripts/train_cv.py` on ConvNeXt-T or EfficientNetV2-S at 384 px (the compute budget is in `kaggle_runbook.md`), 3–5 folds, with L2+L3+L5 on. Save the OOF predictions. | Gold-58 ≥ 0.90 (in the frontier's single-model band) |
| Oct 5–7 | **L4 A/B** on a 2,000-study subset, 3 epochs: `crop_mm` 140 vs 120, same seed. | Keep 120 mm if hold-out macro improves by ≥ 0.003 and no finding drops by > 0.01 |
| Oct 7–11 | **Round 1 (L1):** `scripts/refine_labels.py` on round-0 OOF, then retrain. | Gold-58 Synovitis/Effusion/Contusion up with macro not down; primary metric up |
| Oct 11–15 | **L6:** blend the new pipeline (rank mean, equal weight) with the public CoAtNet family. LB submission. **Accept the rules by Oct 15.** | Keep if LB ≥ base (0.939) |
| Oct 15–19 | Second seed or backbone *only if* runtime allows. Distil the efficiency student (L8). | — |
| Oct 20–22 | Freeze. Pick two final submissions: (a) best LB, (b) most principled (equal weights, no LB-probed choices) to hedge the private shake-up. | — |

## 7. What is implemented in this repository

`src/kneemri/` is an original, unit-tested implementation of the pipeline above. See the
`README.md` for commands.

* **DICOM → canonical volumes** (`volume.py`, `series.py`).
  * Slot selection from `*_series.csv`.
  * Slice sorting by position along the slice normal.
  * Orientation standardisation from direction cosines.
  * Laterality from tags with a geometric fallback, and right→left mirroring.
  * Physical-scale resampling, with optional foreground re-centring and robust intensity
    normalisation.
  * Per-study `.npz` cache.
* **Labels** (`labels/`).
  * Teacher tables: loading, gold-leak detection, gold agreement, fusion.
  * An offline open-weight LLM labeler with a three-state-plus-silence JSON schema.
  * Silence-aware refinement with OOF image predictions (L1).
* **Model** (`model.py`): a timm 2.5-D backbone, then slot + slice-position embeddings, then
  a window-context transformer, then per-finding query cross-attention, then a
  label-interaction layer, then per-finding logits (L2, L3).
* **Training** (`train.py`).
  * Report-hash-grouped folds.
  * Soft BCE plus pairwise AUC loss (L5).
  * EMA, AMP and gradient checkpointing.
  * Gold-58 guard metrics and OOF export.
* **Inference** (`infer.py`, `ensemble.py`).
  * Offline, time-budgeted inference with a safe 0.5 fallback.
  * Fold and model rank-mean, and optional blending with an external submission (L6).

Everything that does not need Kaggle's DICOMs or GPU is exercised by `pytest` on synthetic
DICOM series.

**Nothing here has been scored on the real data yet.** The expected gains in §5 are
research-based estimates, to be confirmed by the §6 protocol.

## References

* Bien N. et al. Deep-learning-assisted diagnosis for knee MRI: MRNet. *PLOS Med* 2018. https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699
* Dunnhofer M. et al. Improving MRI-based knee disorder diagnosis with pyramidal feature details. *MIDL* 2021. https://proceedings.mlr.press/v143/dunnhofer21a.html
* Astuto B. et al. Automatic deep learning–assisted detection and grading of abnormalities in knee MRI studies. *Radiology: AI* 2021. https://pubs.rsna.org/doi/full/10.1148/ryai.2021200165
* Zhao R. et al. fastMRI+: clinical pathology annotations for knee and brain MRI. *Sci Data* 2022. https://www.nature.com/articles/s41597-022-01255-z
* Ilse M., Tomczak J., Welling M. Attention-based deep multiple instance learning. *ICML* 2018. https://arxiv.org/abs/1802.04712
* Shao Z. et al. TransMIL. *NeurIPS* 2021. https://arxiv.org/abs/2106.00908
* Liu S. et al. Query2Label. 2021. https://arxiv.org/abs/2107.10834
* Ridnik T. et al. ML-Decoder. *WACV* 2023. https://arxiv.org/abs/2111.12933
* Irvin J. et al. CheXpert. *AAAI* 2019. https://arxiv.org/abs/1901.07031
* Smit A. et al. CheXbert. *EMNLP* 2020. https://arxiv.org/abs/2004.09167
* Mukherjee P. et al. Feasibility of using the privacy-preserving LLM Vicuna for labeling radiology reports. *Radiology* 2023. https://pubs.rsna.org/doi/full/10.1148/radiol.231147
* Le Guellec B. et al. Performance of an open-source LLM in extracting information from free-text radiology reports. *Radiology: AI* 2024. https://pubs.rsna.org/doi/full/10.1148/ryai.230364
* Reed S. et al. Training deep neural networks on noisy labels with bootstrapping. 2015. https://arxiv.org/abs/1412.6596
* Zhang Z., Sabuncu M. Generalized cross entropy loss. *NeurIPS* 2018. https://arxiv.org/abs/1805.07836
* Han B. et al. Co-teaching. *NeurIPS* 2018. https://arxiv.org/abs/1804.06872
* Li J. et al. DivideMix. *ICLR* 2020. https://arxiv.org/abs/2002.07394
* Liu S. et al. Early-learning regularization. *NeurIPS* 2020. https://arxiv.org/abs/2007.00151
* Song H. et al. Learning from noisy labels with deep neural networks: a survey. *IEEE TNNLS* 2022. https://arxiv.org/abs/2007.08199
* Karimi D. et al. Deep learning with noisy labels in medical image analysis. *MedIA* 2020. https://arxiv.org/abs/1912.02911
* Xie Q. et al. Self-training with Noisy Student. *CVPR* 2020. https://arxiv.org/abs/1911.04252
* Hinton G. et al. Distilling the knowledge in a neural network. 2015. https://arxiv.org/abs/1503.02531
* Vapnik V., Vashist A. A new learning paradigm: learning using privileged information. *Neural Networks* 2009. https://doi.org/10.1016/j.neunet.2009.06.042
* Zhang Y. et al. ConVIRT. 2020. https://arxiv.org/abs/2010.00747
* Huang S.-C. et al. GLoRIA. *ICCV* 2021.
* Boecking B. et al. BioViL. *ECCV* 2022. https://arxiv.org/abs/2204.09817
* Yuan Z. et al. Large-scale robust deep AUC maximization (LibAUC). *ICCV* 2021. https://arxiv.org/abs/2012.03173
* Sanders T. et al. Bone contusion patterns of the knee at MR imaging. *RadioGraphics* 2000. https://pubs.rsna.org/doi/abs/10.1148/radiographics.20.suppl_1.g00oc19s135
* Hunter D. et al. Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS. *Osteoarthritis Cartilage* 2011.
* Akinci D'Antonoli T. et al. TotalSegmentator MRI. *Radiology* 2025. https://pubs.rsna.org/doi/full/10.1148/radiol.241613
* Public competition study of the 0.94 ensemble (experiment ledger, gold-58 panels): https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection
* Public measurements repo (0.935 LB, ensemble and diversity notes): https://github.com/ShivenKhurana1/rsna-knee-abnormality
