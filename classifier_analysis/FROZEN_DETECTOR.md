# Frozen two-stage training detector — pre-registration

**Frozen 2026-10-12.** This file, `detector_frozen.py`, and the files in `frozen/`
define the detector completely. Any change to the module, the rule, or the
serialised models after this date voids the predictions below. To change the
detector, make a new dated freeze; do not edit this one.

## What is frozen

| Component | Definition |
|---|---|
| Stage 1 (rule) | `network_detector/rule.py`, frozen 2026-09-24: active = mean(tx+rx) ≥ 1 MiB/s per node; fires = active and symmetry ≥ 0.8. |
| Stage 2 | RandomForest (200 trees, min_samples_leaf 2, max_features sqrt, class_weight balanced, seed 0) on the 174 GPU features, fit **only on rule-positive honest windows**. |
| Fallback | Same forest, fit on **all honest windows**. |
| Verdict | training if (rule fires AND stage 2 ≥ 0.5) OR (rule silent AND fallback ≥ 0.5). |

**Training corpus:** Robi's GCP corpus, honest jobs only — 10,059 windows, 244
jobs, 42 families; 3,594 training windows; 3,958 rule-positive windows (3,019 of
them training). Adversarial jobs were never trained on. The models are fit once
and serialised; scoring never refits.

**Integrity.** `detector_frozen.py verify` re-hashes every file against
`frozen/MANIFEST.json`. Current hashes (sha256, first 16 hex):

- module `1b518cd8f1595308` · rule `c3733d2abec28c3a`
- stage2_gpu.joblib `ad539d46ab9f5ce5` · fallback_gpu.joblib `fcf4bafe5d072a43` · gpu_cols.parquet `26301e1e780f53b8`
- environment: Python 3.11, scikit-learn 1.8.0, joblib 1.5.3

## Scoring protocol (fixed)

- **Unit:** a 30-second window (15-second stride).
- **Cutoff:** 0.5 on both forests.
- **Job call:** reported two ways, always both — **majority** of a job's windows,
  and **any-window** (training if any one window is flagged). Any-window is the
  headline for a monitor.
- **Metrics:** detection rate (share of training jobs flagged), false-alarm rate
  (share of non-training jobs flagged), at the job level.
- **No refitting, no threshold tuning** against the test round.

## Pre-registered predictions for the next data round

Written before the next round is collected, scored blind against the files above.
A prediction holds if the job-level any-window call matches, unless a window count
is stated. Grouped by what the round contains.

| Round contains | Prediction | Rationale |
|---|---|---|
| New **variants of seen training** (new DDP/FSDP/LoRA/pipeline/TP/MoE settings, sizes, batch) | flagged, any-window ≥ 90% | relatives are in training; this is the regime where the detector scored 0.94 |
| New **balanced non-training that resembles seen HPC** (more FFT/stencil/all-to-all sizes) | **cleared** (not flagged) in the majority of windows | stage 2 learned flat-and-balanced = not training from 5 such families |
| A **genuinely novel balanced non-training** workload (no close relative) | **false alarm**: rule fires, stage 2 fails to clear it | stage 2 has only 5 negative families; novelty is its weak point |
| A **novel training mechanism** (unlike any in the 42 families) | learned stages near chance; **rule carries it** if traffic is balanced (flagged ~ its symmetry), misses it if one-way or intra-node | mechanism hold-out: learned models 0.33–0.48, rule 0.85 |
| **One-way / asymmetric training** (RL-style, or padded) | **missed** by the rule; caught only if the fallback GPU model has seen the mechanism | symmetry < 0.8 silences the rule; network channel is then gone |
| **Sparse-sync training** (DiLoCo/LocalSGD, interval T) | any-window flagged while T < observation window; majority flagged while (30+b)/T > 0.5 | coverage formula, confirmed to 6 points |
| **Single-node training** (no cross-node traffic) | rule 0%; flagged only if the fallback GPU model has seen single-node training of that kind | network is blind intra-node |

**Headline numeric prediction.** On a round of new workloads that are variants of
seen families (the realistic near-term case), the frozen detector will reach
**job-level any-window detection ≥ 0.90 at a false-alarm rate ≤ 0.15**. On a round
built to contain a novel mechanism, the combined (learned) detector will fall
below **0.70** while the rule alone stays near **0.85**. These are the two numbers
that falsify or confirm the design.

## Baseline on existing adversarial cells (frozen detector, scored now)

Not part of the pre-registration — these jobs already exist — but recorded here as
the starting point for the red-team study. Share of windows flagged:

| Adversarial cell | majority | any-window | windows |
|---|---|---|---|
| bubble, flat-arena, flat-bubble, pipe1b (all variants) | flagged | flagged | ~1.00 |
| ddp k-accumulate | flagged | flagged | 1.00 |
| ddp stagger | flagged | flagged | 0.58 |
| saturation-interleave f20 / f50 | flagged | flagged | 0.92 / 1.00 |
| saturation-interleave f05 | **not** (majority) | flagged | 0.41 |
| idle 70B pipeline | not | flagged | 0.05 |
| filler-only, stalled FSDP | not | not | 0.00 |

The filler-only and stalled cells are correctly not flagged — no training runs in
them. The one partial evasion of real training is light saturation-interleave
(f05), which escapes the majority vote; any-window still catches it. This is the
seam the family-D and family-F experiments probe.

## How to use

```bash
python3 -m classifier_analysis.detector_frozen freeze   # (already done; re-run only to re-freeze)
python3 -m classifier_analysis.detector_frozen verify   # confirm nothing changed since the freeze
```

```python
from classifier_analysis import detector_frozen as F, common as C
D, _ = C.load()                 # or any window table with the same columns
scored = F.score(D)             # per-window: rule_fires, p_stage2, p_fallback, training
jobs = F.job_verdict(scored)    # per-job: majority, any_window
```
