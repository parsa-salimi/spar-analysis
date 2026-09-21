# STATUS — what exists and why

**Read this first. Two minutes.** Updated as things land; changelog at the bottom.

**Where to read what**

| you want | read |
|---|---|
| what any of the jargon means | `GLOSSARY.md` |
| how the power analysis works, from scratch | `power_analysis/METHOD.md` |
| what it found | `power_analysis/README.md` |
| how the comm model works, from scratch | `comm_model/METHOD.md` |
| what it found | `comm_model/README.md` |
| why the trace schema exists and what it pins down | `trace/README.md` |
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

## The pieces

| # | Piece | What it answers | State | Where |
|---|---|---|---|---|
| 1 | **Power analysis** | How much can any claim from this corpus be trusted? | ✅ done | `power_analysis/` |
| 2 | **Comm model** | Is there a signal at all, and in which quantity? | ✅ done | `comm_model/` |
| 3 | **Trace schema + generator** | What does the collector write, and how do we build a detector before the hardware works? | 🔨 in progress | `trace/` |
| 4 | Aggregator + features + classifier | The detector itself | ⬜ not started | — |
| 5 | Evasion evaluation | What does hiding cost the adversary? | 🟡 modelled, not measured | `comm_model/evasions.py` |

Each piece depends on the one above it. 1 is the ruler. 2 says what to measure.
3 is the pipe everything else runs through.

---

## What we know so far

**From the power analysis (1):**

- The published 98% is a property of how the data is split, not of the detector.
  Held out honestly it is **0.715 ± 0.080** (leave-one-family-out, n=75), or
  **0.520 ± 0.150** by mechanism — below the 0.68 you get by always guessing
  "training".
- Held out entirely, **LLM inference scores 0.067** — 93% of it is called
  training. False accusation is our failure mode.
- **Validation cannot rank models here.** Picking the best of ten splits by
  validation does *worse* than picking blind.
- **Precision is bought with workload families, not GPU-hours.** 7.2× more
  windows leaves the error bar unchanged. ±0.10 needs ~113 families; we have 75.

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

## Next

- Measurement-artifact study: how slowly can we sample, and how badly can the
  clocks disagree, before detection degrades? Answers what Long's agent needs
  **before** he builds it.
- Then the aggregator and the detector.

---

## Waiting on other people

| Question | Who | Why it matters |
|---|---|---|
| InfiniBand or Ethernet between the two current nodes? | Will / Long | Changes line rate 4×, and the counter-wrap floor with it |
| Is my compute restriction the hardware class or me personally? | Will | Decides whether the CPU-only split is a phase or permanent |
| Ownership tasks written into the plan | Will | Still `[TODO]`; will default if left |
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
