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
python3 -m classifier_analysis.robust           # tuned thresholds, drop-group retraining
python3 -m classifier_analysis.robust_summary   # -> results/robust_workload.csv
python3 -m classifier_analysis.perstream        # per-GPU-stream scoring check
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

Three regimes:

- **Run-grouped (the paper's headline split).** Every channel reaches 0.93–0.96,
  because the split lets a model recognise a workload it has already seen under
  another run id. The interconnect channel is redundant with GPU telemetry here.
- **Workload-held-out (Robi's stricter split).** The channels separate; section
  2a says exactly how.
- **Family-held-out.** Every *learned* model collapses, the interconnect ones
  included: NIC-learned reaches TPR 0.91 but at FPR 0.72. Only the **unfitted
  rule** holds (0.96, FPR 0.002). With a family held out, the training set has no
  example of that mechanism, so "balanced ⇒ training" cannot be learned; it has to
  be encoded.

### 2a. What the network adds on the workload split

The 0.5-threshold table above mixes two different things: whether a model
*ranks* held-out windows correctly, and whether its 0.5 threshold still sits in
the right place for a workload it has never seen. `robust.py` separates them.
Each outer fold picks its threshold by an inner grouped-by-cell CV on the
training cells only; AUC and TPR at a fixed FPR need no threshold at all. Three
seeds, RandomForest and XGBoost, 95% intervals from resampling cells
(`results/robust_workload.csv`):

| features | AUC | balanced acc., 0.5 threshold | balanced acc., tuned threshold | TPR at 1% FPR |
|---|---:|---:|---:|---:|
| NIC only | 0.96 | 0.93–0.95 | **0.93–0.96** [0.87, 1.00] | **0.91–0.92** |
| Everything | 0.95–0.97 | 0.83–0.94 | 0.91–0.95 [0.86, 1.00] | 0.45–0.67 |
| GPU, all 174 | 0.94 | 0.72–0.77 | 0.83–0.92 [0.73, 0.98] | 0.49–0.57 |
| GPU, no interconnect | 0.94 | 0.71–0.84 | 0.81–0.86 [0.65, 0.97] | 0.26–0.32 |

Ranges span the two model types. Read together with the per-cell results:

1. **Detection is nearly the same everywhere.** With tuned thresholds, most
   feature sets detect ~92% of held-out training windows (DDP ResNet-50 ~82%,
   FSDP ~100%). The exceptions are two GPU-only models that miss half of one
   training cell: FSDP-1B for XGBoost on all 174, DDP ResNet-50 for the forest
   without interconnect.
2. **The differences are mostly false alarms.** GPU telemetry mistakes some
   pipeline-sharded serving for training: 25–88% of `tp4pp4` windows and 22–53%
   of `kv32k_out2k` windows, depending on the model. Adding the network's balance
   features removes most of this (`tp4pp4` 33% → 4% for the forest). This is the
   symmetry mechanism doing its job inside the learned model.
3. **GPU telemetry has the ranking, not the threshold.** GPU-only AUC is 0.94,
   close to NIC's 0.96. Its large drop at 0.5 is mostly a threshold that does not
   transfer to unseen workloads. Adding network features makes that threshold
   transfer: "Everything" at 0.5 gains 11–17 points over GPU-only, but only 3–8
   points once thresholds are tuned.
4. **At strict false-alarm budgets, NIC alone is best, and adding GPU features
   hurts.** TPR at 1% FPR is 0.91–0.92 for NIC only against 0.45–0.67 for
   Everything. Most of that gap is one job: `s16_nonml_comm`, a collective loop
   whose GPUs sit at 100% utilisation while the network is quiet in 93% of
   windows (median 0.003 MiB/s). Every model with GPU features calls it training
   in 98–100% of windows; NIC only calls it 7% (the 7% are the active windows,
   which are perfectly balanced). The network is right here because nothing is on
   the wire, not because it understood the job.
5. **The corpus cannot pin the size of these effects.** With 5 training cells and
   16 non-training cells, the cell-bootstrap intervals overlap for every
   comparison above. Direction is consistent across both models and all three
   seeds; magnitude is not established.

**Drop-group retraining** (same protocol): taking the NIC balance group out of
Everything lowers the forest's tuned balanced accuracy from 0.948 to 0.897, and
GPU plus the balance group alone recovers 0.940. For XGBoost, removing balance
makes no difference (0.913 → 0.927), because volume and timing features
substitute for it. The balance group is the network's main contribution in the
forest, and the effect depends on the model.

**Per-stream check** (`perstream.py`, `results/perstream_check.csv`): scoring
each GPU stream separately and taking the median probability, as Robi's code
does, gives the same picture. Workload split: GPU-only 0.75 (RF) / 0.89 (XGB) at
0.5, AUC 0.97–0.98; Everything 0.95 / 0.96. Family split: GPU-only RF TPR 0% at
FPR 75%, Everything RF 16% / 82%, which reproduces `network_detector/README.md`.

### 2b. Answer: how much does the interconnect data matter?

- **On workloads the model has seen** (run split): not at all; every channel
  works.
- **On unseen workloads** (workload split): network features don't add
  detections, they remove false alarms. They stop GPU telemetry from mistaking
  pipeline serving for training, and they make the decision threshold transfer.
  A model trained on everything leans on them for exactly that, and network-only
  is as good as everything, or better at strict false-alarm budgets.
- **On unseen families**: no learned model works, with or without network
  features. Only the mechanism rule does.

---

## 3. Feature importance

Permutation importance on held-out data, RandomForest on Everything: the drop in
balanced accuracy when one feature, or a whole physical group of features, is
shuffled. It is measured at the 0.5 threshold, so it partly measures the
threshold transfer described in 2a; the drop-group retraining in 2a is the
cleaner test.

**Grouped, workload-held-out** (`results/importance_grouped_workload.csv`):

| group | drop in balanced accuracy |
|---|---:|
| **NIC balance** (sym, burst_sym, dir_abs, pair_sym, anti_dir) | **0.112** |
| PCIe | 0.073 |
| Memory | 0.064 |
| NIC volume | 0.030 |
| NIC timing (cv, acf, fft, xcorr) | 0.003 |
| Power & thermal | 0.002 |
| Compute (util, clocks, tensor) | ≈ 0 |

The NIC balance group comes first by a wide margin, and the top five single
features are all network ones (`vol`, `dir_abs`, `sym`, `burst_sym`,
`pair_sym`), ahead of any GPU feature
(`results/importance_single_workload.csv`).

**Grouped, run-grouped** (`results/importance_grouped_run.csv`): every
importance is below 0.01. When the test workloads were seen in training, the
signal is redundant across channels, so no single group matters.

Family-held-out importance is not reported: each held-out family contains one
class only, so balanced accuracy is undefined per fold.

---

## Caveats

- Seven honest families, two of them training; 21 honest cells, 5 of them
  training. Family-held-out numbers rest on one or two families, and
  workload-held-out differences are within cell-bootstrap uncertainty.
- One TCP gVNIC per node, not a production fabric; the 0.956 symmetry ceiling is
  a TCP-header artifact and an RDMA fabric should read closer to 1.0.
- The comparison is learned models vs an unfitted rule. It shows the mechanism
  generalises where fitting on seven families cannot, not that NIC telemetry
  beats NVML for a detector trained on the families it will actually meet.
- DiLoCo is reconstructed from two real corpus phases plus window arithmetic, not
  a measured DiLoCo run. The next collection round can test it directly.
