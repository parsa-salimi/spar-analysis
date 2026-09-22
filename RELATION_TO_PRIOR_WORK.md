# Relation to prior work

**Where this repo sits relative to the paper and to prior analysis of the corpus.**
Parsa Salimi, September 2026.

Read this before quoting any number here against a number from the paper. The
two often answer different questions, and the difference is the grouping level.

---

## 1. What the paper establishes

Rahman & Tajdari, *Detecting Hidden ML Training With Zero-Overhead Telemetry*
(arXiv 2606.19262).

**The 98.2% is in-distribution and grouped by run.** From the footnote to §5.1:

> RandomForest, 30 s windows, stride 15 s, 5-fold grouped CV, 26,496 windows
> across 1,396 runs that produced at least 5 NVML samples.

and Table 4's caption:

> Runs are randomly allocated to a single fold; all windows from the same run
> are within the same fold, so data from a run never appears in both train and
> test data.

The paper labels the result for what it is: *"In-distribution pooled
classification is essentially solved for non-adversarial workloads."*

**The abstract reports generalisation alongside it:**

> We develop a classifier that achieves 98.2% binary accuracy at identifying
> training workloads across the whole corpus, and 43–87% accuracy against the
> most challenging unexpected workloads even when they are adversarially
> disguised.

**Several hold-out protocols, not one:**

| protocol | result |
|---|---|
| 5-fold grouped CV, grouped by run | 98.21% ± 0.84% |
| leave-one-**workload**-out (benign) | 97.6% average |
| leave-one-**GPU**-out | 99.4–100% |
| leave-one-**strategy**-out, 20 evasion families | 59–100%; white-box LoRA < 50% |

**The limitations section scopes three things this repo works on.** Verbatim:

- *"our experiments were conducted exclusively on single-node hardware
  configurations… Future experiments should expand to greater GPU counts,
  including multi-node clusters. Evasion characteristics for multi-node model-
  and data-parallel training will differ due to different types of interconnect
  (such as InfiniBand between servers vs. NVLink within servers)"* — this is our
  project.
- *"our telemetry was generally sampled at ∼1 Hz… Future work should test other
  sampling rates, measuring how much classification improves with more frequent
  sampling, how much sampling can be done without degrading performance, and
  investigating the minimum sample rate that is still difficult to evade."* —
  this is `trace/METHOD.md`.
- *"we only tested a limited range of workloads… notable gaps including
  multi-user inference serving"* — adjacent to the held-out-inference result in
  `power_analysis/`.

---

## 2. What prior analysis of the corpus establishes

Robi Rahman, in the team thread, answering Long Yi on LOO versus train/val/test:

> You can't split at the level of individual runs because then the detector is
> trained on one run from that strategy, memorizes what the evasion looks like,
> and regurgitates the label during testing. You have to hold out a whole family
> to see if the detector works on it.
>
> Perhaps the best path here is to use coding agents to greatly increase the
> number of workloads, so that we have enough for train/val/test splits.

and, from an overnight analysis of the multi-node corpus:

> That collapses 102 workloads to 21 families… Across all 84 (seed, model) runs
> the val–test correlation is zero: Spearman +0.012 (p=0.917), Pearson −0.053.
> Selecting on val gave 0.5688 mean test; always using the plainest config,
> rf200, would have given 0.6039… 21 families means ~4 per test split, so the
> split — not the model — dominates the variance (0.28 to 0.86).

So the following are **established prior results**, not contributions of this
repo:

- workload-level hold-out leaks; whole families are the right unit
- ~21 families in the multi-node corpus
- validation accuracy cannot rank models at this sample size
- selecting on validation actively degrades test accuracy
- the split, not the model, drives the 0.28–0.86 spread
- the corpus has many GPU-seconds but few distinct workload types
- the remedy is to generate more workloads

One methodological caveat that belongs on the record: that analysis and this one
were both produced with Claude, over the same corpus. Agreement between them is
weaker evidence than agreement between two independent analysts would be —
shared priors, shared method. It is reassuring about the arithmetic rather than
about the judgement.

---

## 3. What this repo adds

All of it is of the form "that result, quantified or stress-tested". None of it
is a correction to the paper.

### 3a. The GPU-seconds do not buy precision — measured

Two facts sit side by side in the prior analysis: ~21 distinct families, and
*"millions+ of GPU-seconds"* from *"300 wall-clock hours of telemetry"*. Whether
the GPU-seconds help is the natural objection to acting on the conclusion.

Re-running **identical splits with identical seeds** at three per-run window
caps answers it:

| windows per run | total windows | mean accuracy | sd across splits |
|---|---:|---:|---:|
| 3 | 2,612 | 0.748 | 0.134 |
| 10 | 8,046 | 0.736 | 0.146 |
| 30 | 18,694 | 0.727 | 0.160 |

A 7.2× increase in windows leaves the spread unchanged — it rises slightly. Were
windows independent observations, the standard deviation should have fallen by
√7.2 ≈ 2.7, from 0.134 to about 0.05.

### 3b. A collection budget

"Generate more workloads" is the right direction; the operational question is how
many. Writing SE = √(p(1−p)/n), measuring the variance inflation over independent
Bernoulli families empirically (×1.44) and inverting:

| target 95% CI half-width | families required |
|---|---:|
| ± 0.20 | 28 |
| ± 0.15 | ~52 |
| ± 0.10 | 113 |
| ± 0.05 | 451 |

75 families by the definition in `families.py`, 21–22 by a stricter one — roughly
±0.12 and ±0.21 respectively.

### 3c. A grouping ladder rather than a single level

Five nested levels on one corpus, so the dependence on where the line is drawn is
visible and a reader can locate their own preferred definition on it:

| held out | groups | accuracy |
|---|---:|---:|
| run — *the paper's protocol* | 1,000 | **0.977** |
| workload label | 144 | 0.874 |
| family | 75 | 0.724 |
| mechanism | 22 | 0.523 |

The mechanism level (22 groups, 0.523) lands close to the prior analysis's family
level (21 groups, ~0.60) despite a separately written family map — an independent
check on the shakiest component of either analysis.

### 3d. An interval rather than a point estimate

Leave-one-family-out, with the confidence interval taken from the measured spread
of per-family accuracies and no distributional assumption: **0.715 ± 0.080**
(n = 75 families), **0.520 ± 0.150** (n = 22 mechanisms).

This also answers Long's question about LOO versus train/val/test quantitatively.
Validation is not merely weak here — it is *negatively* correlated with test
(ρ = −0.14 at family level, −0.35 at mechanism level), and picking the best of ten
splits by validation scores 0.711 against 0.721 for picking blind.

### 3e. A false-positive mode on benign workloads

Held out **entirely**, LLM inference scores 0.067 — 93% of it classified as
training. Held-out non-training families average 0.577 against 0.829 for training
families. Neither the paper's protocol nor the prior analysis surfaces this,
because sibling inference workloads remain in the training set in both.

This speaks to Long's other question — whether to model attacks that manufacture
false positives by making inference look like training. On this corpus the
classifier already does that to itself, with no adversary present.

### 3f. Sampling rate, clock skew and counter width

`trace/METHOD.md` answers the paper's limitation 5 and produces a spec line for
the collection agent: sample at ≥ 2 Hz, use 64-bit counters, ordinary NTP is
sufficient.

### 3g. Whether interconnect volume separates the classes at all

`comm_model/` is new work rather than an extension: a closed-form communication
model over 107,128 configurations showing that byte volume separates training
from co-located serving at AUC 0.51, while tx/rx symmetry takes a
held-out-model classifier from 0.76 to 0.97. This determines what the collection
agent must record and has no counterpart in the prior work.

---

## 4. How to report numbers from this repo

- **Always state the grouping level next to the number.** "0.72" means nothing
  without "held out by workload family".
- **Do not use 98.2% as a foil.** It is in-distribution, grouped by run, and
  reproduced here at 0.977. Put the stricter numbers beside it as a different
  measurement, not as a correction.
- **Do not print a fold-to-fold standard deviation as a generalisation
  interval.** The ±0.84% in the paper is the spread across five CV folds, which
  bounds a narrower question than a governance reader will take it to mean. The
  interval for generalisation to unseen workload families is ±0.08 to ±0.15. This
  applies to our own numbers first.
- **Position as extending.** The paper establishes single-node detection and
  names multi-node interconnect as the open problem. That is the project.

---

## 5. One open question for the authors

§6.2 states:

> our classifier learned to recognize AllGather and ReduceScatter NVLink traffic
> associated with forward/backward steps in FSDP training at 7B–70B scale

while §6.4 lists NVLink throughput among the DCGM counters that *would* supply
additional signal in a layered monitor, and the nine NVML features do not include
an NVLink counter — the interconnect-adjacent features are PCIe transmit and
receive, which do not carry intra-node NVLink traffic. In the team corpus the
DCGM NVLink fields have never returned a real value: of 2,050 telemetry files, 76
carry those columns and two hold a value, which is 2⁶³−1, the "not supported"
sentinel.

The likely reading is that §6.2 means the power and utilisation *signature* of
those collectives rather than a byte counter. Worth confirming, because this
project depends on knowing whether NVLink byte counters have ever produced usable
data on this cluster.
