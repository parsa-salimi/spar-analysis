# Classifier analysis — how much the interconnect signal is worth

Three questions, one corpus (Robi Rahman's two-node `multinode-tpu-experiments`
runs: 2 × 8 H100/A100, one NIC per node, per-GPU NVML telemetry and per-node NIC
byte counters recorded together). Honest jobs only, unless stated; adversarial
jobs are never trained on. Unit of analysis: one 30 s window of one job (15 s
stride), the same windows for every detector.

Reproduce (from `spar-analysis/`, with the NIC + GPU window tables exported into
`classifier_analysis/data/` by `export.py`):

```bash
python3 -m classifier_analysis.export          # once, needs the GPU-monitoring clone
python3 -m classifier_analysis.ablation        # the feature-set x model x protocol sweep
python3 -m classifier_analysis.summarize        # -> results/ablation_*.csv
python3 -m classifier_analysis.importance       # run-split grouped + single importances
python3 -m classifier_analysis.diloco_coverage  # the DiLoCo window-coverage table
```

Headline tables are in [`results/`](results/).

---

## 1. Does DiLoCo defeat the detector?

**Short answer: only at long synchronisation intervals, and only against a
job-level majority vote. An any-window rule catches it until the sync interval
grows past the whole observation window; the real price DiLoCo pays is model
quality, not detection.**

DiLoCo (Distributed Low-Communication training) alternates two phases that both
already exist, measured, in the corpus, so no traffic had to be synthesised:

- **Inner steps** — each node trains alone, nothing crosses the network.
  `s16_train_ddpstagger_k` is exactly this: 16 GPUs busy, NIC at 0.02–0.17 MiB/s,
  below the 1 MiB/s activity floor. The rule calls it training in **0%** of
  windows.
- **Outer sync** — an all-reduce of the pseudo-gradient. Every real all-reduce
  job in the corpus (FSDP, DDP) reads per-node symmetry **0.95–0.96** and is
  caught in **~100%** of its active windows.

So a window with a sync is caught; a pure inner-step window is not. Whether
DiLoCo evades is therefore a question of **window coverage**: what fraction of
30 s windows contain an outer sync, as a function of the sync interval `T`. That
is arithmetic (`results/diloco_coverage.csv`):

| sync interval | windows containing a sync (burst) | any-window call | majority-window call |
|---|---:|:--:|:--:|
| ≤ 30 s | ~100% | caught | caught |
| 60 s | 50% | caught | **evades** |
| 120 s | 24% | caught | **evades** |
| 300 s | 9% | caught | **evades** |
| 1200 s | 2% | caught | **evades** |
| ≥ 1 h (one run) | 0% | **evades** | **evades** |

- If the pseudo-gradient is **spread** (trickled across the interval rather than
  burst), every active window is balanced, so coverage is 100% at every interval
  and the rule catches it always. Bursting is the attacker's better move.
- Longer bursts only help the detector (they cover more windows); the table uses
  the pessimistic instantaneous-burst case.
- **Verdict.** Report DiLoCo as caught by an **any-window** rule up to sync
  intervals near the observation window, and as the detector's structural blind
  spot only when syncs are rarer than the entire window of observation. At that
  point the evasion is real but so is the cost: DiLoCo-style infrequent
  synchronisation trades communication for convergence, and that trade is the
  number worth quoting from the DiLoCo literature.

---

## 2. Interconnect-only vs GPU-only vs everything

Balanced accuracy of the best model per feature set (RandomForest / XGBoost /
Logistic, the paper's configurations), under three hold-out protocols. The
unfitted symmetry **rule** is shown for reference (`results/ablation_*.csv`).

| feature set | run-grouped | workload-held-out | family-held-out |
|---|---:|---:|---:|
| NIC (14 network features) | 0.96 | 0.96 | 0.59 |
| NIC without volume | 0.96 | **0.96** | 0.58 |
| PCIe | 0.93 | 0.72 | 0.21 |
| Interconnect (NIC + PCIe) | 0.96 | 0.94 | 0.33 |
| GPU, no interconnect (144) | 0.95 | 0.84 | 0.44 |
| GPU, all 174 (Robi's) | 0.96 | 0.77 | 0.31 |
| Everything (188) | 0.96 | 0.94 | 0.27 |
| **symmetry rule (unfitted)** | **0.96** | **0.96** | **0.96** |

Three regimes, three lessons:

- **Run-grouped (the paper's headline split).** Everything works: every channel
  reaches 0.93–0.96, because the split lets a model recognise a workload it has
  seen under another run id. Here the interconnect channel is neither necessary
  nor special — it is simply redundant with GPU telemetry.
- **Workload-held-out (Robi's stricter split).** The interconnect channel is
  what generalises. NIC-only holds at 0.96 while GPU-only falls to 0.77–0.84;
  "Everything" (0.94) is carried by its NIC features, not its 174 GPU features.
  Dropping byte **volume** *raises* NIC accuracy (0.91 → 0.96 for the forest),
  because absolute volume on this one-gVNIC cluster is a workload fingerprint
  that does not transfer — evidence for reporting results without `vol`, as
  `netfeat.py` already flags.
- **Family-held-out.** Every *learned* model collapses, the interconnect ones
  included: NIC-learned reaches TPR 0.91 but at FPR 0.72 — it fires on
  everything. Only the **unfitted rule** survives (0.96, at FPR 0.002). Learning
  from a handful of families cannot teach "balanced ⇒ training" when the
  held-out family is the only unbalanced negative; the mechanism must be encoded,
  not fitted.

**Answer to "how impactful is the interconnect data":** decisive for
generalisation. On the realistic cross-workload split it carries the whole
result and matches using everything; adding the 174 GPU features on top of it
buys nothing. But no learned feature set, interconnect included, generalises to
an unseen *family* — that is what the rule is for.

---

## 3. Feature importance

Permutation importance (drop in balanced accuracy when a feature, or a whole
physical group, is shuffled on held-out data), RandomForest.

**Grouped, workload-held-out** (`results/importance_grouped_workload.csv`) — the
split where generalisation is actually tested:

| group | importance |
|---|---:|
| **NIC balance** (sym, burst_sym, dir_abs, pair_sym, anti_dir) | **0.112** |
| PCIe | 0.073 |
| Memory | 0.064 |
| NIC volume | 0.030 |
| NIC timing (cv, acf, fft, xcorr) | 0.003 |
| Power & thermal | 0.002 |
| Compute (util, clocks, tensor) | ≈ 0 |

The NIC **balance** group is the single most important signal by more than 2×,
and the top five individual features are all network
(`vol`, `dir_abs`, `sym`, `burst_sym`, `pair_sym`) ahead of any GPU feature
(`results/importance_single_workload.csv`). This is the symmetry mechanism
showing up as the thing the model actually leans on once it can no longer
memorise workloads.

**Grouped, run-grouped** (`results/importance_grouped_run.csv`): every
importance is tiny (max 0.008, Memory). Under the easy split the signal is so
redundant across channels that removing any one group barely moves accuracy —
which is why that split cannot tell you what matters.

---

## Caveats

- Seven honest families, two of them training: every family-held-out number
  rests on one or two families.
- One TCP gVNIC per node, not a production fabric; the 0.956 symmetry ceiling is
  a TCP-header artifact and an RDMA fabric should read closer to 1.0.
- The comparison is learned models vs an unfitted rule. It shows the mechanism
  generalises where fitting on seven families cannot, not that NIC telemetry
  beats NVML for a detector trained on the families it will actually meet.
- DiLoCo is reconstructed from two real corpus phases plus window arithmetic, not
  a measured DiLoCo run. The next collection round can test it directly.
