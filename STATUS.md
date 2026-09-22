# STATUS — what exists and why

**Read this first. Two minutes.** Updated as things land; changelog at the bottom.

**Where to read what**

| you want | read |
|---|---|
| how these numbers relate to the paper's — **read before quoting one against the other** | `RELATION_TO_PRIOR_WORK.md` |
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
| 4 | Aggregator + features + classifier | The detector itself | ⬜ not started | — |
| 5 | Evasion evaluation | What does hiding cost the adversary? | 🟡 modelled, not measured | `comm_model/evasions.py` |

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
- **Detection is coverage, not transfer.** Trained on one training mechanism plus
  all negatives, no mechanism teaches any other: DDP -> FSDP 0.23,
  FSDP -> DDP 0.35, every other pair at or near zero. Yet the union of several
  mechanisms covers DDP at 0.978. The classifier learns a patchwork, not an
  abstraction, and a held-out mechanism is detected only if it lands inside some
  other mechanism's patch.
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
- **tx/rx symmetry is what works**: 0.757 → 0.965. Training averages gradients,
  and averaging is symmetric; serving pushes cached state one direction.
- **Message size would be better still and is unobtainable** from a byte
  counter. It is excluded.
- **Blind spot:** single-GPU training and replicated serving both emit nothing.
- **Three of four evasions are free.** DiLoCo is 1,500× quieter at no cost and
  *faster* when the run was network-bound.

**From the trace work (3):**

- **A 32-bit InfiniBand counter wraps in 0.34 s at line rate.** The existing
  collector samples at 1 Hz, so it rolls over ~3× between readings.
- **The obvious safety check does not work.** Beyond one wrap the wrong answer
  looks plausible: in test, 13.9% of bytes recovered, no alarm raised. The guard
  has to be a configuration check made *before* collection (`preflight()`), not a
  check on the values.

**From the measurement-artifact study (3b):**

- **Spec line for the agent: sample at ≥ 2 Hz, use 64-bit counters, ordinary NTP
  is fine.** Counter width matters more than sample rate across the whole range.
- Below 1 Hz detection degrades sharply; at 0.2 Hz with 32-bit counters it is
  *worse than chance*.
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
4. Nothing from an in-process hook enters the feature set — that is the adversary
   reporting on itself.

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

- The detector itself: aggregator, full feature extractor, and a two-stage
  classifier whose second stage is blind to absolute byte volume.
- Oct 12 midterm outline. The threat model, methods and evaluation sections are
  largely written across the three METHOD documents.

---

## Waiting on other people

| Question | Who | Why it matters |
|---|---|---|
| InfiniBand or Ethernet between the two current nodes? | Will / Long | Changes line rate 4×, and the counter-wrap floor with it |
| Ownership tasks written into the plan | Will | Still `[TODO]`; will default if left |
| Accept ±0.22, knowing more nodes will not narrow it? | Will | Bounds every claim in the December report |
| Do DCGM fields 1011/1012 return bytes or a rate, at what width? | Long | Has never returned a real value in our data |
| vLLM with paged KV in the benign suite | Long | Without it the corpus contains no KV traffic at all |

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
