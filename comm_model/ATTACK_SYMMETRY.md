# Attacking the symmetry feature

**A threat model for tx/rx symmetry, and what it costs to defeat it.**
Parsa Salimi, SPAR F26. September 2026.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any unfamiliar term.

---

## 0. Why this has to be written down before it is measured

Symmetry is the only feature we have reason to believe separates training from
serving over an interconnect. Volume is a coin flip (AUC 0.51); symmetry takes a
held-out-model classifier from 0.76 to 0.97. If symmetry is cheap to defeat, the
collection plan is collecting the wrong thing and the December report has no
claim to make.

An evasion result is only worth something if the adversary's capabilities and
costs are fixed in advance. Otherwise the analysis drifts toward whichever
adversary the defence happens to survive. This document fixes them.

---

## 1. The setting

A monitor sits outside the workload and reads interconnect byte counters. For
each node `n` and each analysis window `w` it observes two numbers per link:

- `T(n,w)` — bytes transmitted
- `R(n,w)` — bytes received

It sees nothing else. No message sizes, no peer identities, no process
information — those are either unavailable from a counter or come from inside
the workload, where the adversary controls them.

**The symmetry statistic.**

```
s(n,w) = min(T, R) / max(T, R)   in [0, 1]
```

`s = 1` is perfectly balanced, `s = 0` is strictly one-directional.

**Why the classes differ.** Training averages gradients, and averaging is
symmetric: all-reduce, all-gather, reduce-scatter and all-to-all each require
every participant to send as much as it receives, so `s ≈ 1`. Disaggregated
serving pushes KV-cache state from prefill nodes to decode nodes, which is
strictly directional, so prefill nodes have `s ≈ 0` and decode nodes likewise.
Pipeline parallelism is directional per transfer but symmetric at node level,
because a middle stage receives as much as it forwards.

**Symmetry clears; it does not accuse.** Co-located serving spread across nodes
— pipeline stages or MoE experts on different nodes — is symmetric at node level
too. So asymmetric inter-node traffic points to disaggregated serving, while
symmetric traffic is ambiguous between training and co-located serving. The
attack below is an attempt to fake the signature that clears a workload, and a
detector still needs something other than symmetry — timing — to separate
training from symmetric serving. `detector/` measures both.

---

## 2. The adversary

**Goal.** Run a distributed training job whose observable is indistinguishable
from disaggregated inference serving. Specifically: drive `s(n,w)` below a
target `s*` on every node, for every window.

**Capabilities.**

- May inject arbitrary additional traffic between nodes it controls.
- May choose which nodes send and which receive, and hold that assignment fixed.
- May shape the injected traffic in time.
- May **not** alter the counters. They are read at the fabric or OS level,
  outside the workload's address space. An adversary who can rewrite counters
  has already defeated the monitor and the question is uninteresting.

**Cost.** Wall-clock slowdown of the real training run, expressed as a multiple
of the unpadded step time. Injected bytes are not themselves a cost — bandwidth
the adversary already rents is free to them — but bytes that displace real
traffic are, because they lengthen the step.

This is the right cost metric because the adversary's objective is to train a
model. An evasion that doubles the time to train is a real tax; one that changes
nothing is not an evasion so much as a default setting.

---

## 3. The attack

**Partition padding.** Split the nodes into a sending set `P` and a receiving
set `D`. Inject dummy traffic from `P` to `D` at rate `ρ` per node. Nodes in `P`
gain `ρ` on tx and nothing on rx; nodes in `D` gain `ρ` on rx and nothing on tx.
Both ends move away from balance, and the partition mimics the prefill/decode
split of real disaggregated serving.

**How much padding.** Let `B` be the node's real inter-node rate, which is
symmetric, so tx = rx = `B`. After padding a node in `P` has tx = `B + ρ`,
rx = `B`, giving

```
s = B / (B + ρ)        so        ρ = B (1/s* − 1)
```

| target `s*` | padding required |
|---|---|
| 0.50 | 1 × real traffic |
| 0.20 | 4 × |
| 0.10 | 9 × |
| 0.05 | 19 × |

Hiding is not marginal. Reaching the symmetry of genuine disaggregated serving
means moving roughly ten times the bytes the training job actually needs.

**What it costs.** Per step, the real job moves `B_step` inter-node bytes and
the padding adds `k·B_step` where `k = 1/s* − 1`. All of it must cross a link of
per-node capacity `C`, so the wire time becomes `(1+k)·B_step / C`. The step
cannot finish before its compute does, so

```
step_padded = max( compute_time, intra_time, (1+k)·B_step / C )
slowdown    = step_padded / step_original
```

This is deliberately generous to the adversary. It assumes padding can be
scheduled into the idle gaps between collectives, so it only costs anything once
the *total* per-step traffic exceeds what the link can carry in a step.

---

## 4. What it costs, measured

Over the 48,654 feasible multi-node training configurations in the grid:

| target `s*` | padding | free | median slowdown | p90 |
|---|---:|---:|---:|---:|
| 0.50 | 1× | 51.5% | 1.00× | 2.00× |
| 0.20 | 4× | 37.4% | 2.13× | 5.00× |
| **0.10** | **9×** | **26.4%** | **4.26×** | **10.00×** |
| 0.05 | 19× | 17.1% | 8.52× | 20.00× |

**The attack is expensive, and it is most expensive exactly where it matters.**
Breaking the `s* = 0.10` row down by regime:

| regime | n | free | median slowdown |
|---|---:|---:|---:|
| compute-bound | 30,696 | 41.9% | 1.38× |
| **communication-bound** | 17,958 | **0.0%** | **10.00×** |
| DDP | 16,620 | 42.9% | 1.53× |
| **FSDP** | 32,034 | 17.9% | **6.23×** |
| world ≤ 32 | 13,096 | 38.7% | 1.73× |
| **world ≥ 128** | 25,862 | 20.2% | **6.74×** |

The mechanism is simple: **you cannot pad a link that is already full.** Large
data-parallel training saturates its NICs, so there is no spare capacity to hide
in, and every padded byte displaces a real one.

Read the communication-bound row as arithmetic, not measurement. A configuration
is communication-bound when its link is full, so padding `k` extra bytes per real
byte costs exactly `1 + k` — the 10.00× in that row is 1 + 9. What the grid
contributes is *how many* configurations are communication-bound (37% of the
multi-node training grid), and that is a property of the configuration prior.
The FSDP and world-size rows are informative in the same limited sense: they say
where the saturated configurations sit.

Small compute-bound DDP jobs can pad cheaply. Those are not the jobs
compute-governance regimes are written about.

**For comparison**, DiLoCo makes a run 1,500× quieter at **zero** wall-clock
cost, and faster when the baseline was network-bound. Against that anchor a
median 4.26× slowdown is a serious tax rather than a free lunch.

**And the attack trades one observable for another.** A cluster padding to
`s* = 0.10` moves ten times the bytes its claimed serving workload requires.
Volume alone is a coin flip at separating the classes, but volume that exceeds
what the stated workload could possibly generate is a different and easier
question. Suppressing symmetry manufactures a volume anomaly.

---

## 5. The escalation ladder — what is not yet tested

The numbers above cover the cheapest attack against the simplest defence. Both
sides can do better, and the honest statement of what we know is a position on
this ladder rather than a verdict.

**Attacker moves.**

| | move | cost | status |
|---|---|---|---|
| A0 | none | — | baseline |
| A1 | constant-rate partition padding | measured above | **measured** |
| A2 | burst-matched padding — shape injected traffic to mask the collectives' timing signature as well as their balance | higher: padding must be synchronised to the real collectives rather than run as a background stream | not modelled |
| A3 | topology-aware padding — choose the partition so the imbalance pattern matches a plausible prefill/decode layout | low | not modelled |

**Defender moves.**

| | move | what it would resist | status |
|---|---|---|---|
| D0 | node-level window symmetry | A0 | in use |
| D1 | symmetry of the burst component — separate a steady baseline, which the attacker sets freely, from the burst component, which A1 leaves untouched | A1 | **built and measured** — see below |
| D2 | cross-node burst correlation — every node enters a collective together, so training bursts are synchronised cluster-wide in a way a background pad is not | A1, partly A2 | not built |

**D1 defeats A1.** A1 adds a *constant* stream: the collective bursts underneath
it are unchanged and still symmetric, and only the baseline moves. Measured on
synthetic traces in `detector/`, training against disaggregated serving, with a
one-feature detector fitted on clean data and a decision at P > 0.5:

| defence | training detected, clean | training detected, padded |
|---|---:|---:|
| raw symmetry (D0) | 100% | **0%** |
| burst symmetry (D1) | 100% | **100%** |

A1 therefore costs the adversary a median 4.26× slowdown and buys nothing against
D1. The next rung is A2 — padding shaped to follow the collectives' bursts — which
D1 does not resist by construction, and which is the first move that costs
engineering rather than bandwidth.

On the full population the question is moot for a different reason: detection
there is carried by timing features that A1 does not move, so even the detector
that uses raw symmetry loses almost nothing under attack.

One caveat on byte conservation, which looks like a defence and is not: over the
whole cluster `Σ T = Σ R` always, so padding cannot create net imbalance. But
genuine disaggregated serving conserves bytes too — prefill sends exactly what
decode receives. Conservation does not separate the classes.

---

## 6. Limitations

**The cost model is analytic.** Slowdown comes from the same closed-form model
as everything else in `comm_model/`, with the same standing caveat: it is a
feasibility bound under a stated configuration prior, not a measurement. Real
congestion behaviour could make padding cheaper (if the fabric absorbs it better
than a linear model suggests) or dearer (if contention is worse than linear).

**The grid is a prior, not a sample of the world.** The 26.4% free rate is a
property of how many small and large configurations the grid contains. The
*regime* breakdown is the robust part: comm-bound saturated links leave no room
to pad, and that follows from arithmetic rather than from the grid's
composition.

**One statistic, one window length.** `s` is computed per node per window. A
different aggregation — per link, per node pair, over a longer window — has a
different cost surface.

**The adversary is not adaptive.** A1 is a fixed strategy, not the best response
to a defence it can see. A2 and A3 exist precisely because a real adversary would
not stop at A1.

---

## 7. Reproducing

```bash
cd comm_model
python3 attack_symmetry.py
```

Writes `results_attack_cost.csv` (cost surface by target symmetry) and
`results_attack_regime.csv` (the `s* = 0.10` breakdown). Both tables in §4 come
from those files.
