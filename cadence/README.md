# Cadence: can timing catch training that has been made one-directional?

**A model, not a measurement.** Every trace here is simulated from stated
parameter ranges (`cadence.py`); the output is a feasibility claim and a
sample-rate requirement, to be checked against real 10 Hz traces.

## The question

The lopsided parameter server (`attack_workloads/train_ps_lopsided.py`)
defeats the symmetry rule: each node's traffic runs one way, like a
pipeline-serving stage. What it should not be able to hide cheaply is how a
training loop moves data — compute, then a large transfer, then compute. Online
serving moves many small pieces as requests arrive. This asks whether the
*shape and timing* of one node's byte counter separates them, and at what
sample rate.

Features are scale-free on purpose — duty cycle, variability, autocorrelation
peak, spectral peak share, regularity of burst start times. No absolute volume:
volume separates these classes easily today, but it is what DiLoCo and
compression attack, so a defence resting on it is not a defence against this
adversary.

## Setup

| family | class | what it is |
|---|---|---|
| `ps_step` | training | lopsided parameter server, dense download every step; step 0.25–3 s |
| `ps_diloco` | training | lopsided outer sync every 10–100 steps |
| `pp_serve` | serving | two-stage pipeline, continuous batching, Poisson requests |
| `disagg` | serving | KV cache pushed per request, Poisson |
| `bursty_serve` | serving | pipeline serving under on/off load |
| `static_batch` | serving | offline batch inference: prefill a batch, decode it, repeat — periodic by nature |
| `ps_jitter` | **attacker** | `ps_step` with random idle before each step (costs 1.25–3× throughput) |
| `ps_paced` | **attacker** | download cut into 50–400 randomly sized pieces sent at random times across the step, with or without idle |

40 jobs per family, 240 s each, observed at 1–50 Hz with 1% timing jitter, 30 s
windows at a 15 s stride. Ordinary families are held out one at a time; the
attackers are never trained on and are scored by a model fitted on every
ordinary family.

## Result

One feature carries it: the **autocorrelation peak** at lags of two samples or
more, which is high when traffic arrives in long contiguous bursts. On its own
it generalises across held-out families better than all six features together —
the same pattern as the network rule, where a mechanism-motivated signal beats a
model learned from a handful of families.

| sample rate | AUC | PS every step | PS DiLoCo | jitter attacker | paced attacker | stealth attacker | static batch (false alarm) | online serving (false alarm) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Hz | 0.14 | 0% | 9% | 1% | 0% | 0% | 23% | 0–58% |
| 2 Hz | 0.40 | 0% | 7% | 1% | 0% | 0% | 27% | 0–78% |
| 5 Hz | 0.81 | 97% | 31% | 1% | 2% | 0% | 19% | 0–49% |
| **10 Hz** | **0.97** | **97%** | **84%** | 26% | 1% | 0% | 8% | 0–13% |
| 20 Hz | 0.99 | 95% | 100% | 68% | 0% | 0% | 6% | 0% |
| 50 Hz | 0.99 | 93% | 100% | 85% | 0% | 0% | 9% | 0% |

**What it says.**

- **At 1 Hz, timing is useless or worse.** A training step shorter than the
  sample interval is smeared into a continuous flow, while bursty serving looks
  periodic. Robi's current collector samples at 1 Hz.
- **At 10 Hz, a non-adaptive one-directional trainer is caught** (AUC 0.97;
  online serving false alarms 0–13%, the 13% being on/off bursty load).
  Reliable detection needs about 8 samples per training step: `ps_step` is
  caught 0–11% of the time below 4, 52% at 4–8, and 94–97% at 8 or more. For
  the 1B cells in the run queue (step ~0.6 s) that is ~13 Hz; multi-second
  production steps are covered at 10 Hz. This is the concrete reason for the
  10 Hz collector change.
- **What is detected is burst shape, not rhythm.** Interval regularity alone
  does poorly under family holdout. The signal is that training moves its data as
  large transfers at line rate. The jitter attacker, whose start times are
  irregular, is still caught at high sample rates for that reason.
- **The paced attacker evades completely, at every sample rate, for free.**
  Cutting the download into small, randomly timed pieces spread over the step
  removes the burst shape. The cost is one step of parameter staleness
  (asynchronous parameter server), plus nothing, as long as the link has room —
  and if it has no room, the traffic was continuous anyway.
- **Offline batch inference is a periodic false alarm** (8% at 10 Hz), as is
  serving under on/off bursty load (13%).

## What follows for the network channel

An adaptive trainer can defeat, in turn:

| defence on per-node byte counters | defeated by | cost |
|---|---|---|
| raw symmetry | lopsided parameter server | free at 2 nodes; ≥ ⅓ of GPU nodes at scale |
| burst symmetry | same | same |
| burst shape / cadence (needs ≥ 10 Hz) | pacing the download in random pieces | one step of staleness |
| **volume** | **rare sync + heavy compression (DiLoCo-class)** | **convergence quality — not established** |

What remains is volume: a paced, lopsided parameter server still moves at least a
model's worth of bytes per step — gigabytes per second, against megabytes per
second for serving. Removing that requires training that communicates orders of
magnitude less, which is a question about optimisation, not about monitoring. So
against an adaptive adversary, per-node counters do not detect training as such;
they force an evading run to be extremely communication-efficient. Beyond that
the defender needs other observables: which nodes talk to which (a parameter
server is a hub), or per-GPU telemetry (a worker is running backward passes).

## Limits

- Everything is simulated; step times, transfer sizes and arrival processes
  are drawn from ranges, not measured.
- Logistic regression on per-window features; one window length.
- The paced attacker is modelled, not implemented. A `--pace` option in
  `train_ps_lopsided.py` (asynchronous download over the next step's compute)
  would let the next collection round test it.

## Reproducing

```bash
python3 cadence.py      # ~20 s; writes results/windows.parquet, results/lofo_by_rate.csv
python3 hardening.py    # ~20 min; writes results/hardening.csv
```

## Can a classifier be hardened against all of these?

`hardening.py` adds a traffic-volume feature, a **stealth attacker** (DiLoCo-style
sync every 50–500 steps, outer update compressed 30–300×, trickled out
continuously in random pieces, so its volume is at or below serving's), and
three training regimes. Gradient-boosted trees, so the classifier is not
limited to a linear boundary. Share of windows called training, 10 Hz, shape +
volume features:

| regime | param. server | DiLoCo | jitter | paced | stealth | worst false alarm |
|---|---:|---:|---:|---:|---:|---|
| **unseen**: ordinary families only | 0% | 79% | 53% | 5% | 0% | disaggregated serving 64% |
| **hardened**: every *other* attack in training | 100% | 85% | 77% | 34% | 0% | disaggregated 77%, static batch 60% |
| **memorised**: same attacks, new runs | 100% | 98% | 95% | 98% | 98% | ≤ 3% |

(50 Hz and shape-only are in `results/hardening.csv`; the pattern is the same.)

- **A classifier can learn an attack it has seen** — 95–99% at ≤ 3% false alarms.
  What it learns is that implementation's fingerprint. In this simulation even
  the paced and stealth attackers carry one (how their pieces are sized and
  timed differs from how serving's are); a real adversary changes implementation,
  which is the "unseen" row.
- **Training on past attacks barely helps against the next one**, and makes
  false alarms worse: with every other attack in the training set, the paced
  attacker is caught 34% of the time and the stealth attacker never, while
  honest disaggregated serving is flagged 77% of the time. Teaching the
  classifier that serving-shaped traffic can be training teaches it to call
  serving training.
- **This is the same result as on real data**, one level up: a model learned
  from a handful of families fits what it has seen and does not transfer. Robi's
  per-GPU detector reads 98% in distribution and collapses on held-out families;
  here the held-out family is an attack.

So no: there is no classifier on per-node byte counters that catches every
evasion. An attacker whose traffic reproduces a benign workload's distribution
cannot be separated from that workload by any classifier on the same
observations — it can only be missed, or the benign workload flagged with it.
What survives an adaptive adversary are constraints the attacker cannot shed
without changing how it trains: an all-reduce is balanced, and a training run
must move at least a model's worth of bytes per synchronisation.
