# What I actually did, and why

**A walkthrough of the power analysis — written assuming no prior context.**
Parsa Salimi, SPAR F26. September 2026.

This is the companion to `analysis/power_analysis/README.md`. The README states
results; this states the process, including the judgement calls and the two bugs
I hit. If you only read one, read this one first.

---

## 0. The question in one paragraph

The project we are extending claims it can tell, from telemetry a cloud provider
already collects, whether a GPU is training a model or doing something else. The
published figure is **98.2% accuracy**. If that number is real, hardware-based
verification of compute-use agreements is basically a solved engineering problem.
If it is an artefact of how the data was split, it is not. Our whole project
inherits that claim as its foundation, so I wanted to know which it was before
we build anything on top.

The short answer: it's an artefact. But *saying* that is easy; the work is in
showing it in a way that survives someone pushing back.

---

## 1. What the data is

A **collector** runs on a machine with GPUs and, once per second, records nine
numbers per GPU from NVIDIA's management library:

> GPU utilisation %, memory utilisation %, memory used (MB), power draw (W),
> temperature (°C), SM clock (MHz), memory clock (MHz), PCIe transmit (MB/s),
> PCIe receive (MB/s)

A **run** is one execution of one workload on one GPU, and produces one file.
Run `gpt2_wikitext2` for ten minutes and you get a file with ~600 rows. Run a
LoRA fine-tune for eight hours and you get 28,824 rows.

The repo has **2,166 such files, 35.4 million rows** in total, collected across
eleven different GPU models over several months.

A quick census (`inventory.py`) gave me the shape of it:

| GPU | files | distinct workload labels |
|---|---:|---:|
| H200 | 412 | 62 |
| B200 | 309 | 86 |
| A100-SXM4-80GB | 282 | 123 |
| AMD RX 9070 XT | 274 | — |
| RTX 5080 | 233 | — |
| …7 more | | |

**Decision 1: I used only the A100-80GB / B200 / H200 subset** — 1,003 files,
515,000 rows, 179 distinct workload labels. Reasons: those are the three GPUs
that carry the multi-GPU DDP and FSDP sweeps, they are datacenter parts rather
than gaming cards, and our project is about multi-node datacenter training. The
consumer and AMD files hold 98% of the *rows* but almost none of the workloads
we care about. The grouping argument in this document applies to them unchanged;
the specific numbers would differ.

---

## 2. How a time series becomes something a classifier can eat

A classifier needs rows with a fixed number of columns. A run is a variable-length
time series. The standard move, and the one this repo uses, is **sliding windows**:

- Cut the run into 30-second windows, starting a new one every 15 seconds (so
  they overlap by half).
- From each window, compute summary statistics of each of the nine signals —
  mean, standard deviation, min, max, four percentiles, IQR, range, coefficient
  of variation, skew, kurtosis. That's 13 statistics × 9 signals = 117 columns.
- Add periodicity features: autocorrelation at several lags, the dominant period
  from an FFT, the power at that peak. Training loops are metronomic, so these
  matter.
- Add a few memory-startup features (does memory jump in the first 30 s, the way
  it does when an optimiser allocates its state?).

Total: **174 numeric features per window.** Each window carries a label —
`training` or `not-training` — inherited from its run.

My 1,003 files produced **32,426 windows**.

*This is the crux of everything that follows:* the classifier sees 32,426 rows
and every accuracy number is computed over rows. It is very natural to feel that
32,426 is a large sample. It isn't, and §5 is about why.

### Two bugs I hit doing this

**(a) I destroyed my own timestamps.** To save memory I cast all float64 columns
to float32. Timestamps are epoch seconds — about 1.79 × 10⁹ — and float32 has
roughly 128 seconds of resolution at that magnitude. Every window in a run
collapsed onto a handful of distinct timestamps, and my deduplication step then
silently threw away 87% of the data (32,426 → 4,653 windows). I caught it because
the surviving count was absurd. Fixed by excluding the time columns from the cast.

**(b) The repo's windowing is O(n²).** For each window it re-slices the entire run
with a boolean mask to recompute "features observable up to now". On a 600-row run
nobody notices; on a 28,824-row run it's quadratic. I wrote `fastwin.py`, which
computes the same quantities from numpy prefixes in linear time, and **verified it
produces bit-identical output** — maximum absolute difference 0.0 across every
numeric feature column on a test run. That's worth offering back to the team
regardless of this analysis.

---

## 3. Train/test splits, and what "leakage" means

Standard practice: fit the model on part of the data, measure it on the part it
never saw. The measured accuracy is meant to predict how it will do on data it
has never seen *in future*.

That only works if the held-out part is genuinely unlike the training part in the
way future data will be. If a window from run #47 is in training and *another
window from run #47* is in test, the model can memorise run #47's power signature
and get the test window right without having learned anything about training-vs-
inference. That is **leakage**, and it inflates accuracy.

Everyone working on this already knew to group by run. The repo does it. The
question I asked is whether grouping by run is *enough*.

It isn't, and here's the concrete reason. Look at four of the workload labels:

```
adversarial_L_diluted_2
adversarial_L_diluted_5
adversarial_L_diluted_10
adversarial_L_diluted_20
```

These are not four workloads. They are **one evasion script run four times with a
dilution parameter set to 2, 5, 10 and 20.** If you hold out `_20` and train on
the other three, the model has already seen that exact mechanism. It will do well,
and that tells you nothing about whether it would catch an evasion strategy it had
never seen.

The same is true of `fsdp_7b / 13b / 30b / 70b` (one scale ladder),
`adversarial_K_ddp / _accum_4 / _accum_8` (one accumulation sweep), and so on.

---

## 4. Defining the grouping levels

So I defined five nested ways of grouping the data, from the repo's convention to
the strictest defensible one, and ran *the identical experiment* at each.

| level | what it treats as one unit | count |
|---|---|---:|
| **run** | one telemetry file = one GPU's stream — *the repo's protocol* | 1,000 |
| **job** | one physical execution; the 8 per-GPU streams of an 8-GPU launch merged | 651 |
| **workload** | the nominal label, with `_gpu3` / `_rep2` suffixes stripped | 144 |
| **family** | one *mechanism*, with its knobs collapsed | 75 |
| **mechanism** | the broad class only (`train.ddp`, `infer.llm`, …) | 22 |

The "family" level is the one I argue for, and it's a **judgement call**, so I made
the rules explicit and machine-readable in `families.py`:

- strip rep/sibling suffixes: `_gpu\d+`, `_rep\d+`, `_ext`
- collapse swept parameters: `_N10`, `_accum_8`, `_bs16`, `_seq4096`, `_1h`
- collapse model scale: `_7b`, `_70b`, `_1p5b`, `_fp16`
- collapse parallel degree: `_tp8`, `_dp`

So all seventeen `whitebox_lora_*` labels become one family, and all sixteen
`fsdp_*` labels become one. **179 labels → 75 families → 22 mechanisms.**
The full mapping is dumped to `family_map.csv`, one row per label, so anyone can
disagree with a specific assignment.

I kept all three of `workload` / `family` / `mechanism` precisely *because* it's a
judgement call. If the conclusion only appeared at one level, it would be an
artefact of where I drew the line. It appears at all of them.

(An incidental check: no family ends up containing both training and non-training
labels, so the splits are never ambiguous.)

---

## 5. The experiment

For each of the five levels, 110–180 times:

1. Randomly deal the **groups** — not the windows — into 60% train, 20%
   validation, 20% test. Every window follows its group.
2. Fit a random forest (100 trees) on the training windows.
3. Record accuracy on validation and on test, plus the accuracy on each
   individual test group.

Three splits rather than two because *validation* is what you'd normally use to
choose between models, and *test* is what you'd report. Keeping both lets me ask
whether validation is any use here — see §8.

**One extra decision: I kept at most 10 evenly-spaced windows per run**, cutting
32,426 windows to 8,046. The justification is the ICC figure below, and §9 is the
experiment that checks the decision didn't manufacture the result.

*A note on how this was run, for anyone reproducing it:* I was working through a
shell on the machine holding the repo, where each command is capped at three
minutes and background processes are killed when the command returns. So
everything is chunked and checkpointed to CSV — `power.py` takes a level, a
number of repetitions and a seed, appends its results, and can be run in several
passes that pool afterwards. Not elegant, but it means nothing is lost to a
timeout.

---

## 6. The result

| what is held out | groups | mean accuracy | sd across splits | trivial baseline |
|---|---:|---:|---:|---:|
| nothing (random windows) | — | **0.991** | — | 0.68 |
| run — *the repo's protocol* | 1,000 | 0.977 | 0.010 | 0.681 |
| job | 651 | 0.966 | 0.018 | 0.680 |
| workload | 144 | 0.874 | 0.075 | 0.687 |
| **family** | **75** | **0.724** | **0.139** | 0.634 |
| mechanism | 22 | 0.523 | 0.231 | 0.483 |

Reading this table:

- **Row 1 reproduces the headline.** Split windows at random, ignoring which run
  they came from, and you get 0.991. So the ~98% in the literature is reproducible
  — as a leaky number.
- **"Trivial baseline"** is what you'd score by ignoring the input entirely and
  always guessing the more common class. 68% of windows are training, so that's
  the bar. At the family level the classifier beats it in only **73%** of splits;
  at the mechanism level, **65%**. It is not merely worse than advertised; in a
  meaningful fraction of splits it is worse than a constant.
- **The sd column is the thing to stare at.** It grows from 0.010 to 0.231 while
  nothing about the classifier changes. That spread is not instability in the
  model — it is the measurement getting noisier as the number of independent
  things being measured falls.

---

## 7. Why "number of windows" is the wrong n

Standard result: if you measure a success rate *p* over *n* independent trials,
the standard error of your estimate is

> SE = √( p(1−p) / n )

and the 95% confidence interval is about ±1.96 × SE. Everything hinges on what
counts in *n*.

Windows are not independent trials. Two windows from the same run, 15 seconds
apart, have nearly identical values in all 174 columns. The repo's own notes
measure this as **ICC(1) ≈ 0.98** — the *intraclass correlation*, the fraction of
total variance that sits between runs rather than within them. At 0.98, an
eight-hour run at 1 Hz carries roughly **1.3 windows' worth** of independent
information, not 28,800.

I have a direct confirmation of this that doesn't depend on their figure. When a
family is held out, **53% of families come back either essentially 0% correct or
essentially 100% correct** — the detector doesn't get some of a family's windows
right and some wrong; it decides the family as a unit. That is what "one
observation" means, made visible.

So the right *n* is the number of families: 75, or 22 by mechanism.

**The honest estimator.** Rather than a random 20% test split, I ran
**leave-one-family-out**: hold out each family in turn, train on all the others,
and record how that family scores. Every family contributes exactly one
observation, the whole corpus is used, and the confidence interval comes from the
measured spread of those 75 numbers — no distributional assumption at all:

| estimator | n | accuracy | 95% CI |
|---|---:|---:|---:|
| leave-one-family-out | 75 | **0.715** | ± 0.080 |
| leave-one-mechanism-out | 22 | **0.520** | ± 0.150 |

**And the failures point the wrong way for a monitor.** Held-out *training*
families score 0.829; held-out *non-training* families score 0.577. At the
mechanism level, when LLM inference is held out entirely it scores **0.067** —
93% of LLM inference is called training. For an instrument meant to inform a
reporting threshold, false accusation is the expensive error, and that is the one
it makes.

---

## 8. Why you can't fix this by tuning

The natural response is "fine, we'll tune the model until the held-out number
comes up." That is not available here, and it's worth being precise about why.

Across repeated family-level splits I have both a validation accuracy and a test
accuracy for each split. If validation carried information about test, they'd be
correlated. The rank correlation is **ρ = −0.138** at family level (p = 0.066)
and **−0.353** at mechanism level (p < 10⁻⁴). Not just uninformative — mildly
*anti*-correlated.

Made concrete: take 10 splits, pick the one with the best validation accuracy,
and look at its test accuracy. You get **0.711**. Picking one at random gets
**0.721**. An oracle that could see the answer would get 0.893. **Selecting on
validation is worse than not selecting.**

This is arithmetic, not bad luck. A validation split holds ~15 families; at
p ≈ 0.72 its own standard error is about 0.12. You cannot rank models with a
ruler whose graduations are wider than the differences you're trying to measure.

---

## 9. The check that the result isn't my own doing

I capped windows at 10 per run. If that cap were producing the effect, everything
above would be an artefact of my preprocessing rather than a fact about the
corpus. So I re-ran **identical splits, same random seeds**, at three caps:

| windows per run | total windows | mean accuracy | sd across splits |
|---|---:|---:|---:|
| 3 | 2,612 | 0.748 | 0.134 |
| 10 | 8,046 | 0.736 | 0.146 |
| 30 | 18,694 | 0.727 | 0.160 |

**7.2× more windows; the spread does not shrink.** It rises slightly. If windows
were independent observations, the standard deviation should have fallen by
√7.2 ≈ 2.7 — from 0.134 to about 0.05.

This is the single most useful result in the analysis, because it converts the
claim from a statistical argument into a measured fact. Anyone who wants to
defend window-counting has to explain this table.

---

## 10. What that costs us, in workloads

Inverting SE = √(p(1−p)/n) for a target interval half-width *E*:

> n = p(1−p) · (1.96 / E)²

with one correction. The observed spread between splits is **1.44×** what perfectly
independent families would give — real families differ from each other more than
coin flips do — so I multiply through by that measured factor rather than
assuming independence. At p = 0.715:

| target 95% CI half-width | families needed |
|---|---:|
| ± 0.20 | 28 |
| ± 0.15 | ~52 |
| ± 0.10 | 113 |
| ± 0.05 | 451 |

We have 75. That buys about **± 0.12**.

The operational content: **precision is bought with families, never with hours.**
Running the same DDP job overnight adds windows and buys nothing (§9). Adding a
parallelism strategy, a serving pattern, or a fabric we've never measured buys a
whole observation. That is a different instruction to give the cluster than "run
it longer", and it's the one I'd want settled before the November allocation.

---

## 11. What I did not do, and where this could be wrong

- **The family map is mine.** I wrote the rules; someone else would draw some
  lines differently. That's why `family_map.csv` lists all 179 assignments and why
  three levels of coarseness are reported. The direction is identical at every
  level — only the magnitude moves.
- **One corpus, one classifier, one task.** Datacenter GPUs only, binary
  training-vs-rest, random forest with 100 trees. A different model might do
  better in absolute terms; nothing about that would change the *width* of the
  interval, which is the actual finding.
- **Accuracy is a poor metric at 68% class imbalance.** I report the trivial
  baseline beside every number for that reason, but balanced accuracy and
  per-class rates should replace it in the report proper.
- **Leave-one-family-out trains on 74 of 75 families each time**, so it slightly
  flatters a detector trained on a smaller corpus than ours.
- **I could not verify the teammate analysis referenced in my handoff notes.** It
  reported 0.6039 held-out-family accuracy and a near-zero validation/test
  correlation. Those figures appear in no text file in the repository — I searched
  all of them — so they were presumably shared only in chat. Everything above is an
  independent rebuild. The agreement in direction and rough magnitude is
  reassuring, but it is corroboration, not replication, and it should not be cited
  as though I checked his work.
- **None of this is about our actual signal.** This is nine mature on-GPU NVML
  counters. Our project's signal — interconnect bytes — has never been successfully
  collected even once. Expect these numbers to start worse, not better.

---

## 12. Files

```
analysis/power_analysis/
  README.md          the results, stated compactly
  METHOD.md          this document
  families.py        the grouping rules — read this to argue with the family map
  family_map.csv     all 179 labels and their assignments
  inventory.py       corpus census
  extract.py         windows + features (chunked; see §5)
  fastwin.py         O(n) causal windowing, bit-identical to the repo's O(n²) version
  power.py           repeated grouped splits
  lofo.py            leave-one-family-out
  analyse.py         summary statistics, design effect, required-n
  sensitivity.py     the window-cap check of §9
  figures.py         the four figures
  figures/           f1 ladder, f2 val-vs-test, f3 required-n, f4 per-family
```
