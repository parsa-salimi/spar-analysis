# Evading the network rule — what it costs

**The rule** (`rule.py`): a node's 30 s window is called training if it moves at
least 1 MiB/s and sends about as much as it receives (symmetry ≥ 0.8).

**What it actually measures.** Symmetry is a property of the *all-reduce*, not of
training. Training that averages gradients with an all-reduce is balanced on
every node, whatever the model. Training that moves gradient information some
other way need not be. Every evasion below exploits that gap, and the question
is what each one costs.

To evade, a training job must look either **idle** (below the activity
threshold) or **one-directional** (symmetry below 0.8). Sending garbage both
ways does neither: it keeps the job balanced and it stays caught. It is useful
only for the opposite attack, making an innocent job look like training.

Measured on Robi Rahman's corpus (one TCP NIC per node):

| quantity | value |
|---|---|
| background on an idle node, 30 s mean | p50 0.006, p99 0.19, max 11.6 MiB/s (one-way bursts) |
| reverse traffic of a one-way bulk stream (ACKs, returned tokens) | 4–17% of forward |
| rule's false alarms if the activity threshold drops to 0.1 MiB/s | 0.9–2% of honest windows |

---

## 1. Synchronise rarely (DiLoCo, local SGD)

Nodes train independently and average every H steps. Between syncs, the network
is idle and the rule makes no call.

**Against a majority-of-windows job call**, this works as soon as syncs are more
than about a minute apart, because most windows are then idle.

**Against an any-window job call it does not**, unless each sync is also made
invisible. A sync is visible in a 30 s window when it moves ≥ 30 MiB, and a
ring all-reduce over two nodes moves about 2 bytes per parameter in bf16 each way:

| model | bytes per sync per node | compression needed to fit under 30 MiB |
|---|---:|---:|
| 1B | 2 GB | 64× |
| 7B | 14 GB | 450× |
| 70B | 140 GB | 4,500× |

Switching to any-window calls costs little: on the honest corpus detection is
unchanged (94.7% of jobs) and false alarms go from 2.2% to 4.3% of jobs.

**Hiding under the noise floor.** Instead of bursting, the sync can be trickled
out continuously below the activity threshold. The defender can lower the
threshold to 0.1 MiB/s for 1–2% false alarms, so the trickle must carry under
3 MB per window. A 7B sync compressed 100× (140 MB) then takes ~23 minutes,
which is comparable to a DiLoCo sync interval of 500 steps at ~2 s per step.

**Cost.** Rare synchronisation alone is close to free (the comm model: 1,500×
less traffic at no wall-clock cost). Getting under the floor as well needs
roughly 100× compression of each sync on top of it. Heavy quantisation of
DiLoCo's outer update has been reported to work; 100× on top of H = 500 is at
the edge of what has been shown, and unverified here.

**Verdict.** This is the network channel's structural limit. A training job
that synchronises rarely enough, and compresses hard enough, can keep its
cross-node traffic at background level. The per-GPU channel has to carry this
regime.

## 2. Make it lopsided: compress one direction

Replace the all-reduce with a parameter server: workers upload compressed
gradients (top-k with error feedback, keeping a fraction *c* of the entries)
and download full parameters. Each worker reads symmetry ≈ c + 0.04 (the
acknowledgement floor), so any c ≲ 0.7 passes as one-directional — the
signature of pipeline serving.

**The cost depends on scale**, and this is the useful result:

- **Sharded servers, every node both worker and server**, read balanced again:
  each node sends full parameters for its shard and compressed gradients for the
  others, and receives the mirror image. Lopsidedness requires nodes whose only
  job is serving parameters.
- **Dedicated GPU servers.** A server sends full parameters to W/S workers. For
  its link not to become slower than the ring all-reduce it replaces (≈ 2 × model
  size per node per sync), S ≥ W/2: **at least a third of the GPU nodes stop
  training**. With fewer servers, communication slows by W / (2S).
- **Dedicated CPU servers** avoid idling GPUs but need about half as many CPU
  nodes as GPU nodes, each with a GPU-class NIC, and an optimizer step on CPU.
- **At two nodes the attack is free**: one server, one worker, and the server's
  link carries the same bytes as the ring would. The cluster in this corpus is
  therefore the defender's worst case, not a typical one.

**Verdict.** Cheap on a small cluster, a real tax at scale. The defender's next
move follows from its structure: a worker that receives a large stream and sends
a small one looks like a pipeline-serving stage, but its GPUs are running
backward passes, and its bursts arrive on a fixed step cadence rather than with
request arrivals. Both need a second channel (per-GPU telemetry, or a sample rate
fast enough to resolve steps — the 10 Hz collector change).

## 3. Split the directions across networks

Send forward traffic on one interface and backward on another, so each
interface looks one-directional. Cheap if a second path exists; defeated by
summing a node's interfaces, which a provider can always do. The collector
already records the sum.

## 4. Pad one direction

Add one-way dummy traffic until the node reads unbalanced. Needs ~9× the job's
own inter-node traffic; costs a median 4.26× slowdown over the configuration
grid and up to 10× on links that are already full
(`comm_model/ATTACK_SYMMETRY.md`). Burst symmetry defeats it: padding raises the
floor and leaves the collective bursts balanced.

## 5. Stay on one node

A single-node job puts nothing on the network. Not an evasion so much as a scope
limit: the network channel only speaks about jobs large enough to need several
nodes, which are the ones compute-governance thresholds are written about.

---

## Summary

| attack | defeats | cost to the adversary |
|---|---|---|
| rare sync | majority-of-windows calls | ~free |
| rare sync + ~100× compression + trickle | any-window calls | unverified convergence cost |
| lopsided parameter server | the symmetry test | free at 2 nodes; ≥ ⅓ of GPU nodes, or W/2S slower comms, at scale |
| split across interfaces | a single-interface monitor | nothing, if a second path exists |
| one-way padding | raw symmetry | 4–10× slowdown; defeated by burst symmetry |
| garbage both ways | nothing | — (a framing attack, not an evasion) |
| paced, lopsided parameter server | symmetry and cadence (≥ 10 Hz) | one step of staleness; still moves a model per step |

## Cadence, the next defence, and how it falls

Simulated in [`../cadence/`](../cadence/README.md): with symmetry defeated, can
the shape and timing of one node's byte counter still separate one-directional
training from one-directional serving?

- **At 10 Hz, yes, against a non-adaptive trainer**: AUC 0.97, parameter server
  caught 97%, false alarms 0–13% on online serving and 8% on offline batch
  inference. The signal is burst shape — training moves each step as a large
  transfer at line rate — and it needs about 8 samples per step. **At 1 Hz it is
  worse than chance.**
- **Pacing defeats it completely and for free**: cutting the download into small,
  randomly timed pieces across the step costs one step of parameter staleness and
  is caught ≤ 3% at every sample rate.

What an adaptive trainer cannot remove from per-node counters without changing
how it trains is **volume** — at least a model's worth of bytes per step.
Removing that is DiLoCo plus heavy compression, at a convergence cost. Against
such an adversary, per-node counters do not detect training so much as force an
evading run to be extremely communication-efficient.

## The rule against every attack

The frozen rule (active and raw symmetry ≥ 0.8) is not learned, so it has no
generalisation gap: it fails exactly where the mechanism says it must.

| attack | frozen rule | why |
|---|---|---|
| interleave / saturating-filler interleave (Robi's, real data) | caught at the rate it trains (42–100%) | its training steps still all-reduce |
| DiLoCo, sync every ~100 steps | caught | most windows contain a balanced burst |
| DiLoCo, sync every ~500 steps | missed by a majority-of-windows job call; caught by an any-window call | syncs land in ~30% of windows |
| DiLoCo + heavy compression, trickled below 1 MiB/s | **missed** — no call | never active |
| lopsided parameter server, c ≲ 0.25 | **missed** | symmetry ≈ 3c + 0.04 |
| lopsided + jitter, or paced | **missed** | still lopsided |
| one-way padding | missed by the frozen rule; caught by burst symmetry | padding raises the floor, not the bursts; costs 4–10× |
| split directions across interfaces | caught | the collector sums a node's interfaces |
| garbage both ways | caught | stays balanced |

False alarms: balanced communication that is not gradient averaging (all-to-all,
halo exchange, collective benchmarks).

## Hardening a learned classifier does not close these gaps

Simulated in `../cadence/hardening.py`. Trained on every other attack, a
gradient-boosted classifier catches the paced attacker 34% of the time and the
stealth attacker never, while flagging 60–77% of honest disaggregated and batch
serving. Trained on the same attacks it is tested on, it catches 95–99% at ≤ 3%
false alarms — by learning each implementation's fingerprint, which an adversary
changes. On per-node byte counters, a classifier cannot separate an attacker
whose traffic reproduces a benign workload from that workload; it can only miss
it or flag both.

**What would test these on hardware**, for the next collection round: a GPU
DiLoCo job at H = 100 and 500, with and without 4-bit outer updates; a 2-node
and, if possible, 4-node parameter-server job with top-k gradient uploads at
c = 0.1 and 0.5.
