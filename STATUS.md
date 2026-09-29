# STATUS — what exists and why

**Read this first. Two minutes.** Updated as things land; changelog at the bottom.

**Where to read what**

| you want | read |
|---|---|
| how these numbers relate to the paper's — **read before quoting one against the other** | `RELATION_TO_PRIOR_WORK.md` |
| what Robi's multi-node branch contains, and what its real NIC data shows | `MULTINODE_BRANCH.md` |
| **network-only detection on real multi-node data** | `network_detector/README.md` |
| attack workloads aimed at the network detector, ready for Robi's queue | `attack_workloads/README.md` |
| can timing catch one-directional training, and at what sample rate (simulated) | `cadence/README.md` |
| what to collect in November and what it buys | `COLLECTION_PLAN.md` |
| what any of the jargon means | `GLOSSARY.md` |
| how the power analysis works, from scratch | `power_analysis/METHOD.md` |
| what it found | `power_analysis/README.md` |
| how the comm model works, from scratch | `comm_model/METHOD.md` |
| what it found | `comm_model/README.md` |
| why the trace schema exists and what it pins down | `trace/README.md` |
| how the artifact study works and what the agent's settings must be | `trace/METHOD.md` |
| the schema itself, for whoever writes the agent | `trace/schema/comm_trace_v1.md` |

Every `METHOD.md` is written assuming no background. Every `README.md` states
results and assumes you have read the corresponding METHOD.

---

## The question

A cloud provider can see how many bytes move between the GPUs it rents out. Does
that alone reveal when a customer is **training** a large model rather than just
**serving** one? Compute-governance proposals assume training is detectable. If
it isn't, or if hiding is cheap, those proposals need rewriting.

---

## The standing constraint

**Compute access is permanently unavailable on this project.** Not a phase to be
waited out: every piece of this work has to be buildable, runnable and verifiable
without the cluster, for its whole duration.

Two consequences shape everything below.

**The synthetic pipeline is primary infrastructure, not insurance.** It was built
so the detector could exist before the hardware collected anything. With the
constraint permanent, it is the only instrument available on this side of the
project — which makes the generator's fidelity a load-bearing assumption, and one
that has never been checked against a real trace. A validation protocol for it,
fixed in advance, is therefore part of the work rather than a nicety.

**The trace schema is a permanent interface, not a temporary bridge.** Everything
crossing between this side and the cluster crosses through it, indefinitely. It
is worth the care it has had.

---

## The pieces

| # | Piece | What it answers | State | Where |
|---|---|---|---|---|
| 1 | **Power analysis** | How much can any claim from this corpus be trusted? | ✅ done | `power_analysis/` |
| 2 | **Comm model** | Is there a signal at all, and in which quantity? | ✅ done | `comm_model/` |
| 3 | **Trace schema + generator** | What does the collector write, and how do we build a detector before the hardware works? | ✅ done | `trace/` |
| 3b | **Measurement-artifact study** | What does the *instrument* cost us, and what must the agent's settings be? | ✅ done | `trace/METHOD.md` |
| 3c | **Collection plan** | What do we collect in November, and what precision does it buy? | ✅ done | `COLLECTION_PLAN.md` |
| 4b | **Network-only detection, real data** | Can a provider detect training from NIC counters alone, on unseen families? | ✅ on Robi's corpus; frozen predictions for the next round | `network_detector/` |
| 4 | **Detector v0** | Which quantities carry the decision, and does the padding attack defeat it? | 🟡 v0 on synthetic traces | `detector/` |
| 5 | Evasion evaluation | What does hiding cost the adversary? | 🟡 modelled, not measured | `comm_model/evasions.py` |
| 5b | **Symmetry attack** | Can the symmetry feature be evaded, and at what price? | 🟡 A1 costed and defeated by D1; A2 not modelled | `comm_model/ATTACK_SYMMETRY.md` |

Each piece depends on the one above it. 1 is the ruler. 2 says what to measure.
3 is the pipe everything else runs through.

---

## What we know so far

**From the power analysis (1):**

- The published 98% is in-distribution, grouped by run; grouping by run here
  reproduces it at 0.977. Under the stricter grouping — whole workload families —
  it is **0.715 ± 0.080** (leave-one-family-out, n=75), or
  **0.520 ± 0.150** by mechanism — below the 0.68 you get by always guessing
  "training".
- Held out entirely, **LLM inference scores 0.067** — 93% of it is called
  training. False accusation is our failure mode.
- **Validation cannot rank models here.** Picking the best of ten splits by
  validation does *worse* than picking blind.
- **Precision is bought with workload families, not GPU-hours.** 7.2× more
  windows leaves the error bar unchanged. ±0.10 needs ~113 families; we have 75.

**From the mechanism-transfer analysis (1b):**

- **The collapse across grouping levels is not a data-volume effect.** Across 22
  held-out classes there is no relationship between surviving training data and
  accuracy (rho = -0.09). Randomly subsampling to 4,644 windows keeps held-out
  DDP at 0.974; removing FSDP specifically, at the same 4,644 windows, gives
  0.438.
- **Detection is coverage, not transfer.** The strongest evidence is the union
  result: no single other mechanism gets held-out DDP above 0.35, while several
  together reach 0.978. The single-source matrix points the same way — trained on
  one training mechanism plus all negatives, DDP -> FSDP 0.23, FSDP -> DDP 0.35,
  every other pair near zero — but that design is biased toward zero, since with
  one positive class anything unlike it looks negative. A fairer test trains on k
  mechanisms and holds out the rest, for increasing k; not yet run.
- **Leave-one-mechanism-out still leaks.** Held out alone, DDP scores 0.978; held
  out alongside FSDP, 0.438. Per-class numbers measure how well the rest of the
  corpus happens to cover that class, so the bottom rung of the grouping ladder
  is an upper bound rather than a floor.
- **Run diversity is what buys precision, not window count.** One run per family
  gives 0.674 +/- 0.22 on held-out DDP; three runs give 0.905 +/- 0.03. Extra
  windows inside runs already collected add nothing.
- **This applies to our own interval arithmetic.** The collection plan treats
  families as independent Bernoulli draws. Family outcomes are determined by
  which other families are present, so the quoted interval is optimistic by an
  amount we have not measured.

**From the comm model (2):**

- **Byte volume is a coin flip**: AUC 0.51 against co-located serving. 7B
  training sustains ~25 GB/s; disaggregated serving sustains ~27 GB/s.
- **tx/rx symmetry separates training from disaggregated serving**: 0.757 →
  0.965. Training averages gradients, and averaging is symmetric; disaggregated
  serving pushes cached state one direction. **It clears; it does not accuse**:
  co-located serving across nodes is symmetric too, so symmetric traffic is
  ambiguous and something else — timing — has to separate it from training.
- **Message size would be better still and is unobtainable** from a byte
  counter. It is excluded.
- **Blind spot:** single-GPU training and replicated serving both emit nothing.
- **Three of four evasions are free.** DiLoCo is 1,500× quieter at no cost and
  *faster* when the run was network-bound.
- **Defeating the symmetry feature is not free, and is dearest where it
  matters.** Padding traffic to the directional balance of real disaggregated
  serving needs 9× the job's own inter-node bytes, which costs a median **4.26×
  wall-clock** across the grid. No communication-bound configuration can do it
  free at all, and FSDP at world ≥ 128 pays 6–7×. You cannot pad a link that is
  already saturated. Threat model and cost metric fixed in advance in
  `comm_model/ATTACK_SYMMETRY.md`.

**From the trace work (3):**

- **A 32-bit InfiniBand counter wraps in 0.34 s at line rate.** The existing
  collector samples at 1 Hz, so it rolls over ~3× between readings.
- **The obvious safety check does not work.** Beyond one wrap the wrong answer
  looks plausible: in test, 13.9% of bytes recovered, no alarm raised. The guard
  has to be a configuration check made *before* collection (`preflight()`), not a
  check on the values.
- **The generator integrates counters exactly.** Any fixed integration grid
  renders transfers shorter than one cell as rare, oversized spikes — unbiased in
  total bytes, so byte-recovery tests pass, but small-message serving traffic
  comes out bursty where the model makes it smooth (inter-node cv 0.72 against
  0.02). Every synthetic result in this repo is computed with exact integration.

**From the measurement-artifact study (3b):**

- **Spec line for the agent: 64-bit counters, 10 Hz sampling with 2 Hz as the
  floor, ordinary NTP.** 0.2 Hz fails; 0.5–5 Hz is intermediate and noisy; 10 Hz
  and above is indistinguishable from the best instrument studied. The 64-bit
  requirement rests on the wrap arithmetic — a wrapped counter reports the wrong
  bytes — not on a measured detection loss, which the study does not resolve.
- **Between 0.5 and 50 Hz the AUC curve is decided by one family.** `ddp_7b`, the
  only training family without NVLink traffic, flips between 0 and 1 regardless
  of the knob — the power analysis's coverage failure in miniature. Per-family
  scores and families-correct are reported alongside every AUC for that reason.
- **Clock skew kills the cross-node synchrony feature (0.94 → 0.03) without
  hurting detection** — the other features carry the signal redundantly.
- The study resolves one workload family = 0.083 AUC. Most knobs therefore read
  "no measurable effect", not "no effect".

---

## Decisions this has already forced

1. The agent records **tx and rx separately, per link**. Summing them destroys
   the only feature that survives.
2. The agent records **raw counter values, never deltas**. Unwrapping and unit
   conversion happen in the reader, where they can be redone.
3. **Use 64-bit extended counters.** The standard 32-bit sysfs IB counters are
   unsafe below ~93 Hz on an 8-NIC node.
4. **Sample at 10 Hz, with 2 Hz as the floor.**
5. Nothing from an in-process hook enters the feature set — that is the adversary
   reporting on itself.

---

**From network-only detection on real data (4b):**

- **A two-number rule on NIC counters — traffic present and balanced — catches
  92% of training from held-out families at 0.2% false alarms** on Robi's
  two-node H100/A100 jobs. Learned models on the same held-out families fail on
  either channel: per-GPU catches 5% at 70% false alarms, a learned network model
  74% at 72%. With seven families, learning cannot generalise; the mechanism does.
- **Pipeline-sharded serving is cleared completely**, including the 72B
  long-context configurations the per-GPU detector alarms on.
- **Interleaving training with filler does not hide it from the network**: it is
  caught at the rate it actually trains.
- **The blind spot is training without cross-node synchronisation**, and the
  predicted false alarms are balanced non-ML communication (all-to-all, halo
  exchange). The rule is frozen with predictions for the next collection round.
- **Symmetry measures the all-reduce, not training.** Evading it
  (`network_detector/ATTACKS.md`): rare synchronisation is the structural limit
  and needs ~100× compression on top to get under the noise floor; a lopsided
  parameter server is free on two nodes but costs ≥ ⅓ of GPU nodes, or
  proportionally slower communication, at scale.

**From the detector (4):**

- **Burst symmetry defeats the padding attack.** Training vs disaggregated
  serving: a raw-symmetry detector goes from 100% to **0%** detection under
  padding; a burst-symmetry detector stays at **100%**.
- **On the full population detection is carried by timing.** Cadence and
  cross-node synchrony alone detect 81% of training windows with no false
  positives, and the attack does not move them. That separation is the
  generator's assumption — training periodic and synchronised, serving
  irregular — so validating timing on the first real traces is now the
  load-bearing check.
- **Raw-volume features cause false positives.** The artifact study's detector
  calls 2 of 5 serving families training (40%); restricting to burst-based and
  shift-invariant features cuts that to 2.4%.

---

**From the collection plan (3c):**

- **Two nodes reach all 14 inter-node communication mechanisms.** One node
  reaches none, since there is no inter-node traffic to observe. Nodes three
  through eight add zero new mechanisms: once data-parallel, pipeline and expert
  groups can each cross a node boundary, every combination of collective kinds is
  already available.
- **The binding constraint is therefore neither cluster time nor node count — it
  is the family definition.** Fourteen mechanisms is all this taxonomy contains
  on any amount of hardware. Getting past fourteen requires the second axis,
  evasion strategies, which costs engineering time to write workloads.
- **±0.20 needs 35 families; ±0.10 needs 138.** With ~28 families this supports
  about **±0.22**, before the dependence between families (see 1b) is accounted
  for.
- **Cluster time is not the constraint**: ~21 node-hours for the whole corpus.
- The family definition is fixed in advance — (communication signature, evasion
  strategy) — and recorded per trace so grouping stays auditable.

---

## Next

- **Pre-register the validation protocol**, Tier B first: burst duty cycle,
  interval regularity and cross-node correlation are what the detector now rests
  on. `trace/validate.py` exists and is untested; `VALIDATION.md` does not.
- **Attack rung A2** — padding shaped to follow the collectives' bursts, which
  burst symmetry does not resist by construction.
- **Population:** a second training family without tensor parallelism, so no
  mechanism is a singleton; then re-run the artifact study.
- **Coverage, fairly tested:** train on k mechanisms, hold out the rest, for
  increasing k; and the false-positive side of the transfer matrix.
- Oct 12 midterm outline.

---

## Waiting on other people

| Question | Who | Why it matters |
|---|---|---|
| InfiniBand or Ethernet between the two current nodes? | Will / Long | Changes line rate 4×, and the counter-wrap floor with it |
| Ownership tasks written into the plan | Will | Still `[TODO]`; will default if left |
| Accept ±0.22, knowing more nodes will not narrow it? | Will | Bounds every claim in the December report |
| Do DCGM fields 1011/1012 return bytes or a rate, at what width? | Long | Has never returned a real value in our data |
| vLLM with paged KV in the benign suite | Long | Without it the corpus contains no KV traffic at all |
| Are the 64-bit extended IB counters (`counters_ext`) exposed? | Long | If not, the agent needs ~93 Hz on an 8-NIC node |

---

## Caveats that travel with these numbers

The comm model's figures come from an **analytic grid I designed**, not from
measurements. They are a feasibility bound under a stated configuration prior —
"could a perfect observer tell these apart?" — not a detection rate. Given that
the power analysis watched 0.99 become 0.52 once the evaluation respected the
real unit of replication, that distinction is worth repeating rather than
assuming.

---

## How this repo is kept current

This file is updated in the same commit as the work it describes — piece state,
findings, forced decisions, open questions, changelog. If STATUS disagrees with
the code, STATUS is the bug.

Findings that took explaining go into a `METHOD.md` rather than into a
conversation, so the explanation outlives the chat it was written in.

---

## Changelog

- **2026-09-25** — `cadence/hardening.py`: a classifier trained on other attacks
  does not catch an unseen one (paced 34%, stealth 0%) and flags 60–77% of honest
  serving; trained on the same attacks it catches 95–99% by fingerprinting them.
  `network_detector/ATTACKS.md` gains the rule-versus-every-attack table.

- **2026-09-24** — `cadence/`: simulated. Burst shape separates one-directional
  training from online serving at 10 Hz (AUC 0.97; ~8 samples per step needed)
  and is worse than chance at 1 Hz. A paced parameter server evades it at every
  rate for one step of staleness. Against an adaptive trainer, what per-node
  counters retain is volume.

- **2026-09-24** — `attack_workloads/`: GPU DiLoCo (rare sync; burst or trickle;
  int8/int4) and a lopsided parameter server (top-c upload, dense download),
  written for Robi's queue. 22/22 CPU checks pass, including exact agreement with
  reference implementations; predicted network-rule calls per cell. The
  parameter server is expected to evade the rule outright on two nodes.

- **2026-09-24** — `network_detector/ATTACKS.md`: what evading the network rule
  costs, anchored on measured noise floor (p99 0.19 MiB/s) and one-way reverse
  traffic (4–17%).

- **2026-09-24** — `network_detector/`: network-only detection on Robi's real
  multi-node corpus. The network rule catches 92% of held-out-family training at
  0.2% false alarms, against 5% at 70% for the learned per-GPU model. Six idle
  jobs labelled as training found in the corpus. Rule frozen with predictions in
  `network_detector/PREDICTIONS.md`.

- **2026-09-24** — `MULTINODE_BRANCH.md`: digest of Robi's multi-node branch.
  Real H100 NIC data: every cross-node training cell has per-node symmetry
  0.954–0.961 and every pipeline-sharded serving cell 0.04–0.18. The generator
  models pipeline serving as symmetric and needs its inference `p2p` flow made
  forward-only.

- **2026-09-22** — Detector v0 (`detector/`): baseline/burst split, burst
  symmetry, and the padding attack with its cost paid inside the trace as a
  slowdown. Burst symmetry defeats the attack; on the full population timing
  carries detection. Generator now integrates counters exactly; the artifact
  study was re-run on it, and its spec line is now 64-bit counters and 10 Hz with
  a 2 Hz floor, with the sample-rate curve shown to be dominated by a single
  coverage-marginal family.

- **2026-09-22** — Symmetry attack formalised and costed. Adversary
  capabilities, cost metric and an attacker/defender escalation ladder fixed in
  advance; the cheapest attack (constant-rate partition padding) measured over
  the grid at a median 4.26× slowdown, and free for no communication-bound
  configuration. Next rung is the D1 defence: symmetry of the high-pass
  component, which constant-rate padding leaves untouched.

- **2026-09-22** — Mechanism-transfer analysis. Established that the collapse
  across grouping levels is not a data-volume effect (four controls), that no
  training mechanism transfers to any other, and that detection here is a
  coverage property of the corpus. Leave-one-mechanism-out is an upper bound, not
  a floor. Node-count enumeration: the mechanism set saturates at two nodes.
  `COLLECTION_PLAN.md` revised accordingly.

- **2026-09-22** — Compute access confirmed permanently unavailable. Recorded as
  a standing constraint; the synthetic pipeline is reclassified from insurance to
  primary instrument, which makes a pre-registered validation protocol for the
  generator part of the required work.

- **2026-09-21** — `COLLECTION_PLAN.md`: 14 inter-node mechanisms reachable on
  three nodes against 35 needed for ±0.20; the family definition fixed in
  advance; collection ordered so a partial corpus stays analysable.

- **2026-09-21** — Added `RELATION_TO_PRIOR_WORK.md`: what the paper
  establishes, what prior analysis of the corpus establishes, and what is new
  here. Positioning throughout is now extension rather than correction.

- **2026-09-21** — Measurement-artifact study: sample rate, counter width, clock
  skew and read skew over 12 workload families. Produced the agent spec line.
  Three self-inflicted problems caught by controls along the way (population
  trivially easy; metric nearly binary; ceiling measuring the generator's own
  bundling approximation), plus a window-alignment bug that had the cross-node
  feature reading 0.25 instead of 1.00.
- **2026-09-21** — Explanatory docs: root `GLOSSARY.md`, `comm_model/METHOD.md`
  (how the separability analysis actually works), `trace/README.md`.
- **2026-09-21** — Trace schema v1, synthetic generator, reader, 13 tests.
  Found that the counter-wrap safety check does not work; replaced with a
  preflight configuration check.
- **2026-09-21** — Comm model: 107,128-configuration sweep, evasion costs, three
  figures, write-up. Fixed three self-inflicted problems along the way
  (circular features, physically impossible throughputs, grid artifacts).
- **2026-09-21** — Repository created; both analyses made portable and
  reproducible from a fresh clone.
- **2026-09-20** — Power analysis: five grouping levels, leave-one-family-out,
  required-n curves, window-cap sensitivity check, write-up and walkthrough.
