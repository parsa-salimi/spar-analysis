# Relation to prior work: the paper, and the team thread

**What the paper claims, what this repo reproduces, what it adds — and where I
mischaracterised it.**
Parsa Salimi, 21 September 2026.

I wrote several things in this repo that were unfair to
*Detecting Hidden ML Training With Zero-Overhead Telemetry*
(Rahman & Tajdari, arXiv 2606.19262), and I wrote up as findings several things
that were already established in the team Slack thread. I had characterised the
paper's evaluation from the codebase rather than from the paper, and I had not
read the thread. This file corrects both.

**Two headline corrections.**

**1. The power analysis does not invalidate the paper.** It reproduces the
paper's own number under the paper's own protocol, then asks a stricter question
the paper does not claim to answer.

**2. The core insight was not mine.** Family grouping, the ~21-family count, the
zero validation–test correlation, and the observation that the split rather than
the model dominates the variance were all established by Robi Rahman in the team
thread before I started. What this repo adds is quantification, robustness
checks, and a collection budget — not the discovery.

---

## 1. What the paper actually claims

**The 98.2% is explicitly in-distribution and explicitly grouped.** From the
footnote to §5.1:

> RandomForest, 30 s windows, stride 15 s, 5-fold grouped CV, 26,496 windows
> across 1,396 runs that produced at least 5 NVML samples.

and from Table 4's caption:

> Runs are randomly allocated to a single fold; all windows from the same run
> are within the same fold, so data from a run never appears in both train and
> test data.

So the grouping is by **run**, stated plainly, and the paper labels the result
for what it is: *"In-distribution pooled classification is essentially solved
for non-adversarial workloads."*

**The abstract carries the generalisation number alongside the headline**, which
I had not acknowledged:

> We develop a classifier that achieves 98.2% binary accuracy at identifying
> training workloads across the whole corpus, **and 43–87% accuracy against the
> most challenging unexpected workloads even when they are adversarially
> disguised.**

**They run several hold-out protocols, not one:**

| protocol | result |
|---|---|
| 5-fold grouped CV, grouped by run | 98.21% ± 0.84% |
| leave-one-**workload**-out (benign) | 97.6% average |
| leave-one-**GPU**-out | 99.4–100% |
| leave-one-**strategy**-out, 20 evasion families | 59–100%; white-box LoRA < 50% |

**And the limitations section names three of the things I thought I had found.**
Verbatim:

- *"our experiments were conducted exclusively on single-node hardware
  configurations… Future experiments should expand to greater GPU counts,
  including multi-node clusters. Evasion characteristics for multi-node model-
  and data-parallel training will differ due to different types of interconnect
  (such as InfiniBand between servers vs. NVLink within servers)"* — this is our
  entire project, already scoped by them as future work.
- *"our telemetry was generally sampled at ∼1 Hz… Future work should test other
  sampling rates, measuring how much classification improves with more frequent
  sampling, how much sampling can be done without degrading performance, and
  investigating the minimum sample rate that is still difficult to evade."* —
  this is precisely the measurement-artifact study in `trace/METHOD.md`.
- *"we only tested a limited range of workloads… notable gaps including
  multi-user inference serving"* — adjacent to my held-out-inference finding.

---

## 2. What was already known in the team thread

Robi Rahman (the paper's first author) posted the following before I began, in
response to Long Yi asking about LOO versus train/val/test splits:

> You can't split at the level of individual runs because then the detector is
> trained on one run from that strategy, memorizes what the evasion looks like,
> and regurgitates the label during testing. You have to hold out a whole family
> to see if the detector works on it.
>
> Perhaps the best path here is to use coding agents to greatly increase the
> number of workloads, so that we have enough for train/val/test splits.

and, reporting an overnight analysis:

> That collapses 102 workloads to 21 families… Across all 84 (seed, model) runs
> the val–test correlation is zero: Spearman +0.012 (p=0.917), Pearson −0.053.
> Selecting on val gave 0.5688 mean test; always using the plainest config,
> rf200, would have given 0.6039… 21 families means ~4 per test split, so the
> split — not the model — dominates the variance (0.28 to 0.86).

So the following were **already established** and are re-derivations in my work,
not findings:

- workload-level hold-out leaks; families are the right unit
- ~21 families in the corpus
- validation accuracy cannot rank models here
- selecting on validation actively hurt
- the split, not the model, drives the 0.28–0.86 spread
- the corpus has many GPU-seconds but few distinct workload types
- the remedy is more workloads

One further point of honesty: **that overnight analysis was itself produced by
Claude**, as was mine. Two Claude runs over the same corpus reaching the same
conclusion is weaker corroboration than two independent analysts would be —
shared priors and shared method. The agreement is reassuring about arithmetic,
not about judgement.

---

## 3. What I got wrong

**"The headline 98% is not a property of the detector, it is a property of the
split."** Unfair as written. The paper says it is in-distribution and gives the
out-of-distribution numbers separately, in the abstract. A reader who quotes
98.2% without its companions is misreading the paper; that is not the paper's
error.

**Treating generalisation as unexamined.** It is examined at length — five
rounds of adversarial iteration, a leave-one-strategy-out protocol over 20
evasion families, and a per-family detection chart (their Figure 4). The
question "does this generalise to workloads it hasn't seen?" is the paper's
central question, not a gap in it.

**Implying the authors hadn't thought of the grouping issue.** They had. Their
grouping is stated in a table caption and is correct for the claim they make.

**One further thing worth stating plainly:** the "Robi" whose overnight analysis
my handoff notes referenced is, on the evidence of the author list and the
`classifier/rebuttal_analysis/` directory, **Robi Rahman, the paper's first
author.** The critical analysis I was treating as an outside challenge to the
paper is the author interrogating his own results. My draft Slack post — framed
as "the team's headline number doesn't survive proper grouping" — was therefore
aimed at the person who wrote both the number and the critique. It has been
rewritten.

---

## 4. What survives as a contribution

Six things. None is a correction to the paper, and none is the core insight,
which was Robi's. They are all of the form "that claim, made quantitative or
made robust".

### 4a. A dose–response curve, not a binary correction

Their strictest benign hold-out is leave-one-**workload-label**-out. That still
leaves near-siblings in the training set: hold out `llm_infer_qwen25_7b_fp16`
and `llm_infer_mistral_7b_q4` is still there to learn from. Collapsing swept
parameters and model scales into **families**, and families into **mechanisms**,
removes the siblings too:

| held out | groups | accuracy |
|---|---:|---:|
| run — *the paper's protocol* | 1,000 | **0.977** ← reproduces their 98.2% |
| workload label — *their leave-one-workload-out* | 144 | 0.874 |
| family | 75 | 0.724 |
| mechanism | 22 | 0.523 |

The top row is the point: **under their protocol I get their answer.** The lower
rows are a different, harder question — "can it detect a *kind* of workload it
has never seen?" — which matters for governance and which they do not claim.

Relative to the thread, the increment is the *ladder* rather than any one rung.
Robi's analysis established that family grouping is the right unit. Running five
nested levels on one corpus turns that from a binary correction into a
dose–response curve, so a reader can see how much of the number depends on where
the line is drawn, and can locate their own preferred definition on it. My
mechanism level (22 groups, 0.523) lands essentially on Robi's family level
(21 groups, ~0.60) despite a separately written family map — which is a useful
independent check on the family definition, the shakiest part of either
analysis.

### 4b. From "more workloads" to a number

Robi's conclusion was *"use coding agents to greatly increase the number of
workloads"*. Correct, and the obvious next question is: how many?

Writing SE = √(p(1−p)/n), measuring the variance inflation over independent
Bernoulli families empirically (×1.44), and inverting gives a budget:

| target 95% CI half-width | families required |
|---|---:|
| ± 0.20 | 28 |
| ± 0.15 | ~52 |
| ± 0.10 | 113 |
| ± 0.05 | 451 |

We have 75 by my family definition, 21–22 by the stricter one. That is roughly
±0.12 and ±0.21 respectively. **This is the main increment over the thread**:
the direction was already right; this says what it costs.

### 4c. The GPU-seconds do not buy precision — measured, not argued

The thread leaves two facts side by side: ~21 distinct families, and *"millions+
of GPU-seconds"* from *"300 wall-clock hours of telemetry"*. The unstated
question is whether the GPU-seconds help. They are the natural objection to the
whole argument, and they are what makes "we only have 21 workloads" feel wrong.

Re-running **identical splits, same seeds**, at three window caps answers it:

| windows per run | total windows | mean accuracy | sd across splits |
|---|---:|---:|---:|
| 3 | 2,612 | 0.748 | 0.134 |
| 10 | 8,046 | 0.736 | 0.146 |
| 30 | 18,694 | 0.727 | 0.160 |

A 7.2× increase in windows leaves the spread unchanged — it rises slightly. If
windows were independent observations the standard deviation should have fallen
by √7.2 ≈ 2.7, from 0.134 to about 0.05.

This is the piece of my work I would defend hardest, because it converts the
argument from a statistical claim into a measurement, and it forecloses the one
objection that would otherwise be raised against acting on Robi's conclusion.

### 4e. An interval, not a point estimate

The thread has point estimates (0.6039) and an observed range (0.28–0.86).
Leave-one-family-out, with the confidence interval taken from the measured
spread of per-family accuracies and no distributional assumption, gives
**0.715 ± 0.080** (n = 75 families) and **0.520 ± 0.150** (n = 22 mechanisms).
That is a number with an interval attached, which is what a report needs.

It also answers Long's original question — *"pros and cons of using LOO vs
train/validation/test set split"* — quantitatively rather than qualitatively.
Not merely that validation is weak here: validation is **negatively** correlated
with test (ρ = −0.14 at family level, −0.35 at mechanism level), and picking the
best of ten splits by validation scores 0.711 against 0.721 for picking blind.
Validation-based selection is worse than not selecting.

### 4d. Benign mechanism hold-out surfaces a false-positive mode

Held out **entirely**, LLM inference scores 0.067 — 93% of it classified as
training. Held-out non-training families average 0.577 against 0.829 for
training families. Neither the paper's protocol nor the thread's analysis would
surface this, because sibling inference workloads remain in training in both.

This also speaks directly to Long's other question — *"should we take into
account the attack modes where attackers try to create false positive signals
(i.e. making inference look like training)"*. The answer from this corpus is
that **the classifier already does that to itself, with no adversary present at
all.** An attacker wanting to generate false positives is pushing on a door that
is open.

### 4f. The sampling-rate study is the paper's stated future work

`trace/METHOD.md` answers limitation 5 more or less as written, and produces the
spec line: ≥ 2 Hz, 64-bit counters, ordinary NTP.

---

## 5. How this repo should position itself

**Extending, not correcting.** The paper establishes single-node detection and
names multi-node interconnect as the open problem. That is our project. Where
this repo's numbers differ from the paper's, the difference is the question
being asked, and every claim should say which question it is answering.

Concretely, when we report numbers in the midterm:

- Always state the grouping level alongside the number. "0.72" means nothing
  without "held out by workload family".
- Do not quote 98.2% as a foil. Quote it as what it is — in-distribution,
  grouped by run, reproduced here at 0.977 — and put our stricter numbers beside
  it as a different measurement.
- Carry the family-level interval on our own numbers, and do not print a
  fold-to-fold standard deviation as though it were a generalisation interval.
  That is the actual lesson, and it applies to us before it applies to anyone
  else.

---

## 6. One open question for the authors

§6.2 states:

> our classifier learned to recognize AllGather and ReduceScatter NVLink traffic
> associated with forward/backward steps in FSDP training at 7B–70B scale

but §6.4 lists NVLink throughput among the DCGM counters that *would* provide
additional signal in a layered monitor, and the nine NVML features used do not
include an NVLink counter — the interconnect-adjacent features are PCIe transmit
and receive, which do not carry intra-node NVLink traffic. In the team corpus,
the DCGM NVLink fields have never returned a real value: of 2,050 telemetry
files, 76 carry those columns and two hold a value, which is 2⁶³−1, the "not
supported" sentinel.

Most likely §6.2 means the classifier picked up the *power and utilisation
signature* of those collectives rather than a byte counter, which is a
reasonable thing to mean and a looser thing to say. Worth clarifying, because
our project depends on knowing whether NVLink byte counters have ever produced
usable data on this cluster.
