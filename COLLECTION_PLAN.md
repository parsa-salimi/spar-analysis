# Collection plan for the November allocation

**What to collect on the H200 nodes, in what order, and what precision it buys.**
Parsa Salimi, September 2026. For review before 1 November.

---

## 0. The one thing to decide now

**The hardware ceiling is below the statistical requirement, and that bounds what
the December report can claim.**

Three 8-GPU H200 nodes can produce **14 distinct inter-node communication
mechanisms**. Reaching ±0.20 on a detection rate needs **35 workload families**;
±0.10 needs **138**. Evasion strategies add a second axis and get us to roughly
28 families in total — about **±0.22**.

This is not a scheduling problem. Collecting 28 families three times over at 15
minutes a run is **21 node-hours** out of a month of availability. The binding
constraint is **mechanism diversity, which is bounded by GPU count**, not cluster
time. Asking for more hours would not help; asking for more nodes, or accepting a
wider interval and designing the claim around it, would.

That decision belongs to Will, and it belongs before collection starts rather
than in December.

---

## 1. The target

From the required-families curve in `power_analysis/`, inflated by the measured
×1.44 between-family heterogeneity. Reported at p = 0.5, which maximises the
requirement and is the honest planning assumption given we do not yet know how
well interconnect detection will work:

| target 95% CI half-width | families needed (p = 0.5) | (p = 0.72, optimistic) |
|---|---:|---:|
| ± 0.25 | 22 | 18 |
| **± 0.20** | **35** | 28 |
| ± 0.15 | 61 | 50 |
| ± 0.10 | 138 | 113 |
| ± 0.05 | 553 | 451 |

A claim of the form *"we detect distributed training at X ± 0.22"* is what this
allocation supports. Anything tighter is not available from three nodes.

---

## 2. The family definition, fixed in advance

**This must be settled before collection, not after.** A corpus whose grouping is
decided retrospectively cannot be shown to have been held out honestly — which is
the failure the power analysis diagnosed in the existing corpus. Fixing it now
costs nothing; fixing it later is impossible.

A **family** is a pair:

> **(communication signature, evasion strategy)**

where the *communication signature* is the set of (collective kind, fabric) pairs
the workload produces — computable mechanically from `comm_model/commvol.py`, so
membership is checkable rather than arguable — and *strategy* is `benign` or a
named evasion.

**Within a family, and therefore NOT a new family:** model identity and scale,
batch size, sequence length, precision, learning rate, optimizer, prompt and
output length, request rate. These are knobs. Varying them gives within-family
variance, which we need, but they do not add statistical replicates.

**Across families:** a different set of collectives, a different fabric for the
same collective, or a different evasion strategy.

Recording the signature in each trace's `manifest.json` at collection time makes
the grouping auditable afterwards.

---

## 3. What three nodes can actually produce

Enumerating every configuration feasible on ≤ 24 H200 GPUs (world size ≤ 24,
tensor parallelism ≤ 8 so it stays in-node, memory-feasible with activation
checkpointing) gives **1,403 configurations** collapsing to **33 communication
signatures**, of which **20 involve inter-node traffic** and **14 are distinct
(class, inter-node mechanism) pairs** — the unit that matters, since training p2p
and serving p2p are different families.

| # | class | inter-node mechanism | representative config | nodes | inter GB/s |
|---:|---|---|---|---:|---:|
| 1 | serve | `kv_push` | Llama-2-7B disaggregated | 1–2 | 50.0 |
| 2 | serve | `p2p` | Llama-2-7B colocated tp4 pp3 | 2 | 3.1 |
| 3 | serve | `alltoall + p2p` | Mixtral-8x7B colocated tp4 pp3 | 2 | 33.7 |
| 4 | serve | `kv_push + p2p` | Llama-2-7B disaggregated tp4 pp3 | 2 | 50.0 |
| 5 | serve | `alltoall + kv_push + p2p` | Mixtral-8x7B disaggregated tp4 pp3 | 2 | 42.4 |
| 6 | train | `allreduce` | Llama-2-7B DDP dp3 tp4 | 2 | 3.4 |
| 7 | train | `p2p` | Llama-2-7B dp1 tp4 pp3 | 2 | 1.6 |
| 8 | train | `allgather + reducescatter` | Llama-2-7B FSDP dp3 tp4 | 2 | 6.8 |
| 9 | train | `allreduce + p2p` | Llama-2-7B DDP dp3 tp1 pp3 | 2 | 3.8 |
| 10 | train | `allgather + p2p + reducescatter` | Llama-2-7B FSDP dp3 tp1 pp3 | 2 | 7.2 |
| 11 | train | `allreduce + alltoall` | Mixtral-8x7B DDP dp3 tp8 | 3 | 36.0 |
| 12 | train | `allgather + alltoall + reducescatter` | Mixtral-8x7B FSDP dp3 tp4 | 2 | 36.5 |
| 13 | train | `allreduce + alltoall + p2p` | Mixtral-8x7B DDP dp2 tp4 pp2 | 2 | 15.8 |
| 14 | train | `allgather + alltoall + p2p + reducescatter` | Mixtral-8x7B FSDP dp3 tp1 pp3 | 2 | 27.8 |

Two things to note. Most mechanisms need only **two** nodes, so collection can
start before the third is up. And the inter-node rates span 1.6 to 50 GB/s —
overlapping across classes, which is the comm model's separability result showing
up in the collection plan.

**Configurations that need all 24 GPUs exist but are single instances** — there is
exactly one way to get `allreduce + alltoall` at world 24. They are not a source
of diversity.

---

## 4. The collection list

### Tier 1 — the 14 benign mechanisms (must have)

Rows 1–14 above, three runs each with *different* model, batch and sequence
settings so within-family variance is estimable. **42 runs.** Without this tier
nothing else is interpretable.

### Tier 2 — evasion strategies (the second axis)

Applied to the training mechanisms where they make sense. Each
(mechanism, strategy) pair is a distinct family.

| strategy | applies to | why it is a distinct family |
|---|---|---|
| DiLoCo (H = 100, 500) | `allreduce`, `allgather+reducescatter` | same signature, ~1/H the rate |
| gradient accumulation (k = 8, 32) | all DDP/FSDP | same signature, 1/k the per-token rate |
| traffic shaping (cap 25%, 10%) | any | same signature, rate-limited |
| run segmentation | any | per-step traffic unchanged, run boundary moves |
| KV disguise | `allreduce` | *changes* signature to resemble `kv_push` |
| staggered dilution | `allreduce` | spreads the collective across time |

Roughly **14 additional families** once near-duplicates are dropped. **42 runs.**

### Tier 3 — within-family variation (not new families)

Model scale, batch, sequence, prompt length, request rate. Needed to estimate
within-family variance and to check that the family definition holds — two
configurations in the same family should be classified alike. Fold into Tier 1
and 2 runs rather than collecting separately.

**Total: ~28 families, ~84 runs.**

---

## 5. Time budget

| | |
|---|---|
| runs | ~84 |
| duration per run | 15 min (≈ 56 windows at 30 s / 15 s stride) |
| node-hours | **~21** |
| plus setup, failures, reruns (×2) | **~42** |

Against roughly a month of three-node availability. **Cluster time is not the
constraint and should not be the thing we ask for.** What we need is the
engineering time to write 28 distinct workloads — which is where Robi's
suggestion of using coding agents to generate workloads applies directly.

---

## 6. Collection order

Ordered so that a partial corpus is still analysable, because it probably will be
partial.

1. **Two mechanisms, one training and one serving, end to end** — row 6 and row 1.
   Validates the agent, the schema, the reader and the wrap guard on real
   hardware before anything is collected at scale. If the DCGM NVLink fields still
   return the not-supported sentinel, we find out here rather than in week three.
2. **The rest of Tier 1 benign**, two-node mechanisms first so the third node
   isn't a blocker.
3. **Tier 2 evasions**, DiLoCo first — it is the evasion with no measured cost and
   therefore the one most likely to defeat us.
4. **Three-node mechanisms** (rows 11 and anything requiring world > 16).
5. **Within-family variation** to fill out Tier 3.

After each tier, re-run the leave-one-family-out evaluation. The interval should
narrow as √n; if it doesn't, the family definition is wrong and it is better to
learn that at 14 families than at 28.

---

## 7. What to record

Per `trace/schema/comm_trace_v1.md`, with the settings the artifact study
established:

- **tx and rx separately, per link.** Summing them discards the only feature that
  survives held-out-family evaluation.
- **≥ 2 Hz** sampling.
- **64-bit counters.** Counter width matters more than sample rate across the
  whole range studied; the standard 32-bit InfiniBand word counters wrap in
  0.34 s at line rate.
- Ordinary NTP (≤ 10 ms) is sufficient; PTP is not worth the trouble.
- The **communication signature** in `manifest.json`, so grouping stays auditable.
- Both NVLink and NIC counters, kept separate.

---

## 8. Decisions needed before 1 November

| question | who | why it blocks |
|---|---|---|
| Accept ±0.22, or seek more nodes? | Will | Bounds every claim in the December report |
| InfiniBand or Ethernet between nodes? | Will / Long | Changes inter-node rates ~4× and the counter-wrap floor with it |
| Do DCGM fields 1011/1012 return real values on this cluster? | Long | If not, Tier 1 step 1 fails and the whole plan needs a different NVLink source |
| Are 64-bit extended IB counters exposed? | Long | If not, sampling must rise to ~93 Hz on an 8-NIC node |
| Who writes the 28 workloads? | Will / Long / me | 21 node-hours of compute, but substantial engineering time |

---

## 9. What this plan assumes, and where it could be wrong

- **Every configuration here comes from the analytic model**, not from
  measurement. The mechanism list is robust — it follows from what the
  parallelism strategies do — but the rates assume H200 nodes with 50 GB/s
  per-GPU NICs, and the actual fabric is still unknown.
- **The ×1.44 heterogeneity inflation is measured on the NVML corpus**, not on
  interconnect data. If interconnect families differ from each other more than
  NVML families do, the requirement rises.
- **The 14-mechanism ceiling is specific to 24 GPUs.** It is a property of how
  few ways there are to factor a world size of 24 into dp × tp × pp with tp ≤ 8,
  not a deep fact. More nodes would raise it, though not linearly.
- **Evasion families may be less distinct than counted.** Two strategies that
  produce the same rate pattern are one family for detection purposes even if
  they are two different scripts. If that collapses Tier 2, the achievable
  interval widens further, and it is better to discover that by evaluating after
  Tier 2 than by assuming it now.
