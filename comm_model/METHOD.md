# How the communication model works

**A walkthrough of the separability analysis — written assuming no background.**
Parsa Salimi, SPAR F26. September 2026.

Companion to `comm_model/README.md`, which states results. This states the
method, including the two things I got wrong on the way and had to fix. If a
term here isn't defined, see `GLOSSARY.md` at the repo root.

---

## 0. The question, and why it could be answered without hardware

Our project proposes to detect distributed training by watching how many bytes
cross the interconnect. The project plan puts an early-stopping gate at week 4:
if training and inference turn out not to be separable on communication, we
pivot to writing up a negative result.

That gate was scheduled to depend on cluster data we don't have. But a large
part of it doesn't need data at all, because **the number of bytes a training
step moves is arithmetic**, fixed by the model's shape, the numeric precision,
and how the work is split across GPUs. The same is true of inference. So the
question "do the two overlap?" can be answered by computing both and comparing.

---

## 1. The most important thing to understand: nothing is simulated

There is **no traffic, no time series, no randomness** anywhere in this model.
Each configuration produces a handful of numbers from closed-form formulas. Same
configuration in, same numbers out, every time. It is a large deterministic
lookup table.

(The *generator* — which does produce time series — is a separate piece, in
`trace/`. Confusing the two is easy and I have not always been careful about it.)

A **configuration** is a choice of:

- which model (Llama-3-70B, Mixtral-8x7B, …)
- which GPU (A100-80GB, H100, H200, B200)
- how the work is split: data-parallel × tensor-parallel × pipeline degrees
- batch size, sequence length, gradient accumulation, activation checkpointing
- for training: DDP or FSDP
- for serving: co-located / disaggregated / replicated, plus batch, prompt
  length and output length

---

## 2. One configuration, computed end to end

**Llama-3-70B, FSDP, 8-way data parallel × 8-way tensor parallel on H100 —
64 GPUs, i.e. 8 nodes of 8. Batch 512, sequence 4096, bf16.**

| quantity | how it is obtained | value |
|---|---|---|
| parameters this GPU is responsible for | 70.6B ÷ 8 tensor-parallel ranks | **8.8B** |
| sequences per data-parallel replica | 512 ÷ 8 | 64 |
| activation tensor crossing one TP collective | 64 × 4096 × 8192 × 2 bytes | **4.29 GB** |
| TP collectives per step | 4 per layer × 80 layers | **320** |
| ring all-reduce factor at N = 8 | 2(N−1)/N | **1.75** |
| → tensor-parallel traffic | 320 × 1.75 × 4.29 GB | **2,405 GB per GPU per step**, on NVLink |
| FSDP cross-node traffic | 2 all-gathers + 1 reduce-scatter × 17.65 GB | **46 GB per GPU per step**, on the NIC |
| compute time | 6 × 70.6e9 × 512 × 4096 = 8.9 × 10¹⁷ FLOPs ÷ (64 × 989 TFLOP/s × 0.45) | **31.2 s** |
| communication time | 2,405 GB ÷ 450 GB/s | 5.3 s |
| step time | max(compute, comm) — compute dominates here | **31.2 s** |
| **stored in the table** | bytes ÷ step time | **77.1 GB/s NVLink, 1.5 GB/s NIC** |

Two details in that table are easy to get wrong and both matter:

**The ring factor is 2(N−1)/N, not 2.** A ring all-reduce is a reduce-scatter
followed by an all-gather, each moving (N−1)/N of the buffer per GPU. At N = 2
the factor is exactly **1.0**, so the common shorthand `* 2` over-counts a
two-GPU run by a factor of two. (Long's `nccl_hook` uses `* 2`.)

**The step cannot be shorter than its own traffic takes on the wire.** Taking
`max(compute, comm)` rather than just compute is what stops the table containing
configurations that move more bytes per second than the fabric can carry. It
also produces a result in its own right: **33.9% of the training configurations
are communication-bound** — sitting idle waiting on the network.

---

## 3. The grid

The full cross-product, minus anything that doesn't fit in GPU memory:

| | count |
|---|---:|
| **training** | **52,978** |
| — DDP | 18,298 |
| — FSDP | 34,680 |
| **serving** | **54,150** |
| — co-located | 26,562 |
| — disaggregated prefill/decode | 26,562 |
| — replicated (one model per GPU) | 1,026 |
| **total feasible** | **107,128** |

Per model, from Llama-3-8B (17,340 configurations) down to DeepSeek-V3 (1,978 —
it is 671B parameters and fits almost nowhere).

---

## 4. Two different measurements, and what "trained on what" means for each

This is the part I ran together in conversation and shouldn't have. The two
numbers answer different questions and only one of them involves fitting
anything.

### 4a. The 0.51 — nothing is trained

No model is fitted. It is a rank statistic: pick one training configuration and
one serving configuration at random, and ask how often the training one moves
more bytes. That is all "AUC of a threshold" means. 0.5 is a coin flip. **There
is no train/test split because nothing is learned.**

| comparison | AUC |
|---|---:|
| training vs **co-located serving** | **0.510** |
| training vs disaggregated prefill/decode | 0.588 |
| training vs all serving modes pooled | 0.538 |

And **100% of training configurations lie inside serving's 5th–95th percentile
volume band.**

The reason is a coincidence of scale: 7B training under DDP sustains about
25 GB/s, and disaggregated serving at 25 requests/second with 2048-token prompts
pushes about 27 GB/s of KV cache. Same bytes, entirely different activity.

### 4b. The 0.76 → 0.97 — here a model *is* fitted, and the split is by model family

A logistic regression is fitted on every configuration of **8 models** and
scored on the **9th**, rotating through all nine and averaging. It therefore
never sees DeepSeek-V3 during training and must generalise to it.

This deliberately mirrors the leave-one-family-out discipline from the power
analysis. Without it the classifier could memorise "Llama-3-70B at dp=64 is
training" and score beautifully while having learned nothing.

| what the scorer may use | held-out-model AUC |
|---|---:|
| total bytes/s alone | **0.433** |
| volume split by fabric (total / inter-node / intra-node) | 0.757 |
| **+ tx/rx symmetry** | **0.965** |
| + temporal structure of the rate | 0.978 |
| + message size and collective count | 1.000 |

### 4c. So where is it a coin flip, and where isn't it?

**0.433 is the number to look at.** A *fitted* single-volume model scores **worse
than chance** on an unseen model. Below 0.5 means the direction is inverted: fit
on eight models and it learns "more bytes ⇒ training", which is correct at 7B —
where training moves more than serving — and backwards at 70B, where serving
with large KV caches moves more than training. **There is no volume threshold
that works across model scales.** Not a weak one; none.

**Symmetry is what rescues it**, 0.757 → 0.965. Training averages gradients
across machines, and averaging is symmetric by construction: every participant
sends exactly what it receives. Serving pushes cached attention state one
direction, from the machine that read the prompt to the machine generating
tokens. And transmit and receive are **separate hardware registers**, so the
feature costs nothing extra to collect.

**The last row is a trap, not a result.** Message size scores highest of all and
is **invisible to a cumulative byte counter at any sample rate** — a counter
reports how many bytes crossed, never in what sized pieces. Obtaining it needs
an in-process hook, which is the adversary instrumenting itself. It is excluded
from the feature set and drawn hatched in the figure so nobody quotes it.

### 4d. The blind spot

Single-GPU training and replicated serving both emit **zero** interconnect
traffic. They are indistinguishable, and not because the detector is weak —
there is nothing to observe. This is a scope limit on the whole approach and
belongs in the threat model.

---

## 5. Two things I got wrong

### 5a. I nearly proved my own assumption

The first version attached a symmetry and a cadence value to each workload **by
class**: training got one constant, serving another. Any classifier fed those
separates the classes perfectly, because I had already separated them by hand.
It returned **AUC 1.000**, which is not a result — it is a mirror.

The fix was to attach those properties to the **mechanism** instead. A ring
all-reduce is symmetric because of what a ring all-reduce *is*; a KV-cache push
is directional because it goes from prefill to decode. Derived that way they
genuinely cross-cut the classes: tensor-parallel **serving** is all-reduce
traffic and therefore symmetric, while pipeline **training** is directional.
That is what makes a result from them mean something.

`test_commvol.py` asserts exactly this, so the mistake cannot come back:

```
PASS  colocated TP INFERENCE is symmetric: 1 vs 1
PASS  pipeline TRAINING is directional: 0 vs 0
```

### 5b. Symmetry at transfer level is not symmetry at node level

Building the trace generator forced a correction. `commvol` attaches `sym` to a
**flow**, which is a per-transfer property — one pipeline send really is
strictly one-directional. But a monitor observes **per-node aggregates**, and a
middle pipeline stage receives from its predecessor as much as it sends to its
successor. At node level, pipeline traffic is symmetric.

The classes still separate, but for a narrower and sharper reason than I first
claimed: what is genuinely asymmetric at node level is the **KV-cache push**,
where prefill nodes only send and decode nodes only receive. The generator
derives symmetry from actual accumulated tx/rx rather than from the flow label,
and `trace/test_trace.py` pins the distinction:

```
PASS  inter-node symmetry: training ~1, disaggregated serving ~0  [0.999 vs 0.000]
PASS  ...while that SAME serving trace is symmetric on NVLink     [1.000]
```

### 5c. And one physical error

Before the bandwidth clamp, the grid contained serving configurations at
**716 GB/s per GPU** on hardware whose NVLink tops out at 450. Adding
`max(compute, comm)` fixed it, and produced §2's communication-bound finding as
a side effect.

---

## 6. Verification

`test_commvol.py` pins 20 quantities to numbers derivable by hand. If these fail,
nothing else in this directory means anything. Among them:

- ring factor 1.0 at N=2, 1.75 at N=8, → 2 for large N
- DDP 7B bf16 = 26.9 GB/step/GPU
- FSDP ÷ DDP = exactly 1.5 in bytes per step
- TP at 7B / batch 8 / seq 2048 = 128 collectives × 134.2 MB
- KV cache per token = 512 KiB with multi-head attention vs 128 KiB with
  grouped-query attention — a 4× ratio
- NVLink : NIC = 9 : 1 per GPU
- DiLoCo at H=500 cuts sync volume by exactly 500×

Two of those tests originally asserted on byte **rates** and failed once the
bandwidth clamp went in — because at dp=1024 both DDP and FSDP are pinned at NIC
bandwidth, so they move different volumes at the same rate. The tests now assert
on bytes per step, and separately assert the comm-bound behaviour, which is
itself worth pinning.

---

## 7. The caveat that governs every number above

**This is an analytic grid I designed, not a sample of the world.** Every AUC
here is a property of the configuration prior encoded in the grid — how many
training configurations, how many serving ones, at what scales. It is a
**feasibility bound, not a detection rate**. It answers "could a perfect
observer of these quantities tell them apart?" It does not answer "what will our
detector score on real traffic?"

The power analysis is why this matters. There, a number that looked like 0.99
became 0.52 once the evaluation respected the real unit of replication. The same
held-out-family discipline is applied here, which is why §4b holds out whole
model families. But the deeper lesson transfers: **a clean number on data you
designed is not evidence.** These figures are for deciding what to build and
what to collect.

Other limits worth naming: achieved utilisation varies more in practice than the
0.45 assumed; compute/communication overlap is modelled as perfect, which
flatters throughput and understates communication-bound cases; collectives are
assumed ring-based, while NCCL uses tree algorithms for small messages; and the
serving throughput model is memory-bandwidth-bound decode, ignoring prefill
contention under continuous batching.
