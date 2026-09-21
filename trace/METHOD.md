# What does the instrument cost us?

**The measurement-artifact study — written assuming no background.**
Parsa Salimi, SPAR F26. September 2026.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any unfamiliar term.

---

## 0. The question

Whoever writes the collection agent has to choose three numbers: how often to
read the counters, how tightly to sync the clocks between machines, and which
counters to read. Every one of those choices throws information away
**irreversibly** — you cannot recover a burst you sampled straight through.

So: for each knob, how much detection do we lose, and where is the knee? The
output isn't a graph. It is a line in the schema: *sample at ≥ X Hz, keep clocks
within Y ms, read tx and rx within Z ms of each other.*

## 1. Why it has to be synthetic

On real hardware you only ever observe a workload **through** the instrument. If
detection comes out poor you cannot tell whether the adversary hid well or the
sampling was too coarse — the two are confounded, and no amount of real data
separates them.

Here the true byte counts are generated first and then degraded deliberately,
one knob at a time. The difference is attributable by construction. This is the
one thing the generator buys that the cluster cannot.

---

## 2. Design

**The traffic is built once.** `build_events()` lays down every transfer; each
condition then observes *the same events* through a different instrument
(`sample_events()`). A difference between conditions therefore cannot be a
different random draw — it is the instrument. This is structural, not a
discipline I have to remember.

**The population is deliberately hard** — 7 training families and 5 serving
families, chosen so the classes overlap on the things that are easy to see (§4a
explains what happened when they didn't). Both sides use tensor parallelism
inside the node, so both carry heavy NVLink all-reduce traffic. Both sides
include pipeline-parallel configurations spanning nodes and cross-node MoE
expert parallelism, both of which are *symmetric at node level*. Deliberately
excluded: replicated serving and single-GPU training, which emit nothing and
would be trivially separable from anything that communicates.

**The knobs**, varied one at a time from a reference of 10 Hz / 64-bit / 1 ms
NTP / 80 µs read:

| knob | range |
|---|---|
| sample rate | 0.2 – 50 Hz |
| counter width | 64-bit bytes vs 32-bit 4-byte words |
| clock skew between nodes | 0 – 1000 ms |
| tx/rx read skew | 0 – 10 ms |

**The metric: leave-one-family-out, scores pooled.** Each family is held out
once, its out-of-fold scores collected, and one AUC computed over all of them.
Uncertainty comes from bootstrapping over **families**, which is the actual unit
of replication — the lesson from the power analysis, applied to my own study.

**Three controls**, because a curve with no anchors is unreadable:

| control | value | what it establishes |
|---|---:|---|
| floor — labels shuffled | **0.471** | must be ~0.5, or something leaks |
| hardness — volume features only, at the ceiling | **0.457** | the population really is hard, demonstrated not asserted |
| ceiling — 50 Hz, 64-bit, perfect clocks | **0.902** (11/12 families) | the best instrument we would actually consider |

---

## 3. The result

![figures/a1_measurement_artifacts.png](figures/a1_measurement_artifacts.png)

**The study's resolution is one workload family, which is 1/12 = 0.083 of AUC.**
Nothing smaller than that is measurable here, and most of the differences in the
table are one family. Stating this first is the only honest way to read
everything that follows.

### What is resolved

**Sampling below 1 Hz fails.** At 0.2 Hz with 64-bit counters the AUC falls to
0.572 and three families flip; with 32-bit counters it falls to 0.401 — *worse
than chance*, with only 6 of 12 families correct. At 0.5 Hz the 64-bit case has
already recovered to 0.871 (10/12).

**Counter width matters more than sample rate.** 32-bit word counters are worse
at every rate below 20 Hz, and the gap is large: 0.592 vs 0.873 at 5 Hz, 0.664 vs
0.779 at 10 Hz. This is the wraparound problem from `trace/README.md` biting
exactly as predicted — a 32-bit word counter holds 17.2 GB, which at line rate
is a third of a second.

### What is not resolved

**Clock skew has no measurable effect on detection** anywhere from 0 to 1000 ms.
Every paired comparison against the ceiling crosses zero.

**tx/rx read skew has no measurable effect** up to 10 ms.

**Between 1 Hz and 50 Hz with 64-bit counters, nothing is distinguishable.** The
apparent wobble — 0.779 at 10 Hz, 0.939 at 20 Hz — is one family flipping back
and forth. It is not a finding and should not be read as one.

### The one genuinely interesting result

**Clock skew destroys the cross-node synchrony feature without hurting
detection.** That feature — do all the nodes transmit at the same instant? —
falls from 0.944 at perfect sync to 0.929 at 10 ms, 0.281 at 100 ms and 0.025 at
1 second. It is comprehensively dead. Detection does not move, because the other
features carry the signal redundantly.

Two things follow. Clocks only need to be good enough for the feature we might
want later, not for detection today; and a feature dying is not the same as
detection dying, which is a distinction worth keeping in mind when we start
reading real traces and one number looks wrong.

---

## 4. Three things I got wrong

Each was caught by a control, which is the argument for having them.

### 4a. The first population was trivially easy

Every training family used `tp=1` and every serving family `tp=8`. So no training
family had **any** NVLink traffic and every serving family did — "has NVLink
traffic" separated the classes perfectly. Every serving family's inter-node
traffic was also a directional KV push, so symmetry separated them perfectly too.

Result: **AUC 1.000 in every single condition.** No knob had any effect, because
there was no room for one. I had asserted the population was hard and not
checked. The hardness control (volume-only AUC) now exists precisely so that
claim is measured: it reads 0.457, so volume alone is at chance on this
population.

### 4b. The first metric was nearly binary

I initially held out one training family and one serving family together and
took the mean AUC over the 35 pairs. That metric turned out to be almost
binary — a held-out pair is decided as a unit, so its AUC is 0 or 1, with a
standard deviation of **0.40** across pairs. It left a standard error of ~0.07 on
every condition, far too wide.

This is the same all-or-nothing behaviour the power analysis found for workload
families, appearing again one level up. Pooling out-of-fold scores across folds
fixed it.

### 4c. The ceiling was measuring my own approximation

The original ceiling sampled at 200 Hz. But the generator bundles flows that
fire faster than any sampler could resolve — decode-phase tensor parallelism does
128 all-reduces every 0.72 ms, which over a 240 s trace is 43 million transfers,
intractable to enumerate and invisible at any rate we study. Bundling makes those
flows artificially regular at sub-millisecond scale, and a 200 Hz sampler can
*see* that regularity. The ceiling was measuring my approximation rather than the
workload. It is now 50 Hz, the top of the sweep and safely below where bundling
becomes visible.

A fourth thing, found before the study could run at all: `trace_features` was
joining the per-node and cross-node tables on floating-point window starts
computed from two different time origins. Most rows missed and were filled with
zero, so the cross-node synchrony feature read 0.25 when its true per-window
value was 1.00. Both tables now carry an integer window index off a shared grid.

---

## 5. What goes in the schema

> **Sample at ≥ 2 Hz.** 1 Hz is the point at which nothing is yet measurably
> lost; 2 Hz is margin. Below 1 Hz detection degrades sharply and below 0.5 Hz it
> collapses. Sampling faster than 2 Hz bought nothing measurable here, so the
> cost of going faster is Long's to weigh against agent overhead, not ours.
>
> **Use the 64-bit extended counters.** This is the single most consequential
> choice — larger than the sample rate over the whole range studied. If only
> 32-bit counters are available, the required rate rises steeply and detection
> is degraded at every rate below 20 Hz.
>
> **Ordinary NTP (≤10 ms) is sufficient.** It is enough for detection by a wide
> margin, and it is also what the cross-node synchrony feature needs. PTP is not
> worth the trouble.
>
> **tx/rx read skew up to 10 ms is harmless.** No need to read the two counters
> atomically.

---

## 6. Limitations, and they are real

**This is a lower bound on the required sample rate.** If real traffic is burstier
than modelled, the true requirement is *higher*. It tells Long where to start,
not where to stop, and must be re-validated against the first real traces.

**Twelve families is not many.** The resolution is 0.083 AUC, which is why most
knobs come back "no measurable effect" rather than "no effect". Resolving a 0.03
difference would need roughly ten times the families — the same arithmetic as the
power analysis, and the same answer: precision is bought with families.

**The generator is the ground truth, so its errors are invisible to this study.**
Bundling, the ramp model for transfers, and the assumption that ranks enter a
collective within about a millisecond of each other are all modelling choices.
§4c is one case where a modelling choice leaked into a result and had to be
designed around; there may be others I have not found.

**One detector, one feature set.** Logistic regression on 13 counter-derived
features. A different detector might lean on different features and have a
different sensitivity to the sample rate.

**No cost model for the agent.** Sampling at 50 Hz across a cluster is real CPU
and storage overhead. This study gives only the detection side of that trade.

---

## 7. Reproducing

```bash
cd trace
python3 artifacts.py features 0 26    # generate + cache features per condition
python3 artifacts.py eval             # LOFO, bootstrap, paired comparisons
python3 figures_artifacts.py
```

Outputs land in `artifact_results/`: `conditions.csv` (AUC and CI per
condition), `per_family.csv` (per-family out-of-fold scores), `controls.json`
(floor, hardness, ceiling), `paired_vs_ceiling.csv`, and the figure.
