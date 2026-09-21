# Glossary

Every term this repository assumes, defined once. If something is missing or
unclear, that's a bug — say so and it gets added.

---

## Model shape

**Parameter (Ψ)** — one learned number in the model. "7B" means 7 billion of
them.

**bf16 / fp16 / fp8** — how many bytes each parameter occupies: 2, 2 and 1.
Training is usually bf16 with a higher-precision copy kept for the optimizer.

**Layer (L), hidden size (H)** — a transformer is L near-identical blocks, each
operating on vectors of width H. Llama-3-70B: L = 80, H = 8192.

**Step** — one pass of forward computation, backward computation, and a weight
update, over one batch. Training is millions of these.

---

## Splitting one job across many GPUs

Four independent axes. Real frontier training combines them — typically 8-way
tensor parallel inside a node, data parallel across nodes.

### DDP — Distributed Data Parallel

The simple one. **Every GPU holds a complete copy of the model.** The batch is
split: with 8 GPUs and a batch of 256, each GPU processes 32 examples. Each
computes gradients from its own examples, so they now disagree. To remain one
model rather than eight, they average the gradients — that is the **all-reduce**
— and every GPU applies the identical update.

- **Communication:** one all-reduce per step of the whole gradient buffer,
  **2Ψb bytes per GPU**. Llama-2-7B in bf16: 26.9 GB per GPU per step, in one
  large burst at the step boundary.
- **The catch:** every GPU needs the whole model *plus* its optimizer state.
  Under mixed-precision Adam that is about **16 bytes per parameter** — 2 for the
  weight, 2 for the gradient, 4 for a high-precision master copy, 8 for Adam's
  two running averages. A 70B model needs 1.1 TB of it; an H100 has 80 GB. So
  DDP alone cannot train large models.

### FSDP — Fully Sharded Data Parallel

PyTorch's name for what the literature calls ZeRO-3. The same data-parallel
idea, but it fixes the memory problem: **no GPU holds the whole model.** Each
holds 1/N of the parameters, gradients and optimizer state.

The price is fetching what you don't own, just in time:

1. **Forward:** before computing layer *k*, all-gather layer *k*'s parameters
   from whoever owns them, use them, discard them.
2. **Backward:** all-gather those parameters again, compute gradients.
3. **Then reduce-scatter** the gradients, so each GPU keeps only the averaged
   slice it owns.

Three passes over the parameters rather than two, so **3Ψb per GPU per step —
exactly 1.5× DDP**. The shape differs too, and that matters for detection: DDP
is *one large spike per step*, FSDP is *many small per-layer bursts*.

**The trade in one line:** DDP is cheaper in bytes but needs the model to fit;
FSDP costs 1.5× the bytes and lets you train what doesn't. Above roughly 13B on
80 GB cards you have no choice.

**Why the distinction keeps coming up here:** in the power analysis, held-out
`train.ddp` scored **0.978** and held-out `train.fsdp` scored **0.577**. DDP is
the toy case; FSDP is what anyone training a frontier model actually uses. That
gap is an open research question in this repo.

### Tensor parallel (TP)

Split each individual weight matrix across GPUs, so every GPU does part of every
layer. Costs **4 all-reduces per layer per step**, each the size of an activation
tensor. Bandwidth-hungry, so it is kept inside one node — NVLink is 9× faster
per GPU than the NIC.

### Pipeline parallel (PP)

Put different layers on different GPUs and pass activations along the chain.
Cheap, point-to-point, and directional *per transfer* — though a middle stage
receives as much as it sends, so at **node** level it is symmetric.

### MoE — mixture of experts

Only a fraction of the parameters is used for any given token. Cheap to compute,
expensive to communicate: two all-to-alls per expert layer.

---

## Collective operations

An operation every GPU in a group takes part in together.

| collective | what it does | symmetric? |
|---|---|---|
| **all-reduce** | everyone contributes a buffer, everyone ends up with the sum. How gradients are averaged. | yes |
| **all-gather** | everyone contributes a slice, everyone ends up with all slices | yes |
| **reduce-scatter** | everyone contributes a full buffer, each keeps the summed version of its own slice | yes |
| **all-to-all** | every GPU sends a different piece to every other. Routes tokens to experts. | yes |
| **point-to-point** | one GPU sends to one other | no |

**Ring all-reduce cost.** Implemented as reduce-scatter + all-gather, each moving
(N−1)/N of the buffer per GPU, so the total on the wire per GPU is
**2(N−1)/N × M**, not 2M. At N = 2 the factor is exactly **1.0**.

---

## Inference

**KV cache** — the per-token key/value state a language model keeps so that
generating token *n+1* doesn't require reprocessing tokens 1…*n*. Its size per
token dominates inference memory and inter-node traffic:
`2 × L × n_kv_heads × d_head × bytes`.

**GQA — grouped-query attention** — several attention heads share one key/value
head, shrinking the KV cache several-fold. Llama-2-7B (no GQA): **512 KiB per
token**. Llama-3-8B (8-way GQA): **128 KiB**.

**Prefill / decode** — prefill processes the whole prompt at once and is
compute-bound; decode generates one token at a time and is
memory-bandwidth-bound.

**Disaggregated serving** — prefill and decode run on separate GPUs, so the
prompt's KV cache is **pushed across the network once per request**. This is the
traffic that is directional, and therefore the traffic symmetry catches.

**Co-located serving** — prefill and decode on the same GPUs; only tensor-parallel
traffic, which is all-reduce, and therefore symmetric.

**Replicated serving** — one whole model per GPU. No interconnect traffic at all.

---

## Hardware and measurement

**NVLink** — the fabric between GPUs inside one server. 450 GB/s per GPU per
direction on H100.

**NIC / InfiniBand / RoCE** — the network between servers. 50 GB/s per GPU on
IB NDR — **9× slower** than NVLink, which is why tensor parallelism stays inside
a node and why inter-node traffic is almost entirely the DDP or FSDP sync.

**DCGM** — NVIDIA's management layer; fields 1011/1012 are meant to report
NVLink bytes. They have never returned a real value in our data.

**sysfs counters** — `/sys/class/infiniband/<dev>/ports/<n>/counters/port_xmit_data`
counts **4-byte words**, not bytes, and is **32 bits wide** on the standard
counters. `/sys/class/net/<if>/statistics/tx_bytes` counts bytes and is 64-bit.

**Counter wraparound** — counters fill and roll back to zero. A 32-bit word
counter holds 17.2 GB, which at IB NDR line rate is **0.34 seconds**.

**MFU — model FLOP utilisation** — the fraction of the hardware's peak arithmetic
rate a real run achieves. 0.4–0.5 is normal.

**Activation checkpointing** — discard intermediate activations in the forward
pass and recompute them in the backward pass. ~33% more compute for much less
memory.

---

## Evasions

**DiLoCo** — each worker takes H local steps and workers synchronise only once
per H. Cuts data-parallel traffic by a factor of H. In our model: 1,500× quieter
at no wall-clock cost, and *faster* when the baseline was network-bound.

**Gradient accumulation** — accumulate gradients over k micro-batches before
syncing. Cuts sync traffic per token by k, at the cost of a larger effective
batch.

**Traffic shaping** — cap interconnect utilisation. The bytes still have to move,
so the step simply takes longer.

**Run segmentation** — chop a long run into pieces to stay under a reporting
threshold. Per-step traffic is unchanged; only the run *boundary* moves.

**Compute structuring** — Seferis & Fist's term for splitting a training run
across providers so that no single one sees a reportable run. The threat model
this project inherits.

---

## Statistics

**AUC** — the probability that a randomly chosen positive scores higher than a
randomly chosen negative. 1.0 perfect, 0.5 a coin flip, **below 0.5 means the
score points the wrong way**.

**ICC — intraclass correlation** — the fraction of total variance that sits
*between* groups rather than *within* them. At ICC ≈ 0.98, an 8-hour run at 1 Hz
carries about 1.3 windows' worth of independent information, not 28,800.

**Leave-one-family-out** — hold out each workload family in turn, train on the
rest. The confidence interval comes from the measured spread of per-family
results, with no distributional assumption.

**Window** — a 30-second slice of a telemetry stream, summarised into ~174
statistics. The unit the classifier scores, but *not* the unit of statistical
replication.

**Family** — one mechanism with its knobs collapsed. `adversarial_L_diluted_2 /
_5 / _10 / _20` is one dilution knob turned four times, not four workloads. The
actual unit of replication: 179 workload labels reduce to 75 families.
