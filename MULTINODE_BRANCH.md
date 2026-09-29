# The multi-node branch: what exists, what it shows, what it means here

**A digest of Robi Rahman's `multinode-tpu-experiments` branch of GPU-monitoring,
read 24 September 2026 at commit `2a7d8eb`, and what it changes for this repo.**

See [`GLOSSARY.md`](GLOSSARY.md) for any unfamiliar term.

---

## 1. What was run

| | hardware | scopes | sessions |
|---|---|---|---|
| GPU, multi-node | 2 × a3-highgpu-8g (16 × H100-80GB), GCP | s1, s8 (one node each), s16 (both nodes) | 08-20, 08-21 (adversarial), 09-08/09, 09-17 |
| GPU, multi-node | 2 × a2-highgpu-8g (16 × A100-40GB); A100-80GB pair queued | s16 | 09-04, 09-17, 09-21 |
| TPU | v5e and v6e: single chip, single host, multi-host, multislice over DCN | tpuT1 … tpuT16, tpuTMS | throughout |

**The network is not a production training fabric.** Nodes sit on GCP's default
network with one gVNIC each and no GPUDirect-TCPX, RDMA or InfiniBand, so NCCL
runs over TCP sockets. FSDP saturates it at about 120–140 Gbit/s per direction
per node — 10–25× less than a production H100 fabric. Anything that depends on
how communication-bound a job is (per-GPU power, memory activity, NIC duty
cycle) may be a property of this network rather than of multi-node training.

**Telemetry.**
- Per GPU: NVML/DCGM at ~1 Hz, the 174-feature window set, including DCGM NVLink
  fields 1011/1012 as one aggregate per GPU.
- Per node: a NIC sidecar reading `/proc/net/dev` at 1 Hz, **tx and rx
  separately**, stored as rates in MiB/s. Raw cumulative counters are discarded,
  so counter width and wraparound cannot be checked. Files end at the last 60 s
  checkpoint, so up to a minute is lost per run.
- TPU: libtpu duty cycle, HBM, HLO timing and DCN/collective *latency*
  distributions. There are no interconnect byte counters.

**The NIC channel is not in Robi's classifier.** The loader excludes the
sidecar files; the interconnect appears only as a descriptive observation.

## 2. What is readable here

Every `*.parquet` on the branch is in Git LFS and git-lfs is not installed, so
**the per-GPU multi-node data, the TPU data and most NIC files are not on
disk**. What is readable is 121 NIC files saved as `*.parquet.partial`, which
escaped LFS by their extension: **87 distinct runs, 17 of them with both
nodes**, from the H100 sessions of 08-20 and 08-21. The rest needs
`git lfs pull` on the branch, or access to the GCS bucket.

## 3. What the real NIC data says

Per-node symmetry = min(Σtx, Σrx) / max(Σtx, Σrx), H100 s16, one NIC per node:

| cell | tx (MiB/s) | rx (MiB/s) | symmetry |
|---|---:|---:|---:|
| train_fsdp30b | 16,400–17,400 | 17,100–18,200 | 0.956 |
| train_fsdp1b | 13,600–14,300 | 14,200–15,000 | 0.956 |
| train_ddp_resnet50 | 810–870 | 850–905 | 0.959 |
| train_ddpkaccum8 (gradient accumulation) | 90 | 94 | 0.961 |
| advtrain_interleave_f50, advtrain_oracle | 5,300 / 13,400 | 5,600 / 14,000 | 0.956 |
| infershard_qwen72b_tp8pp2, node 0 / node 1 | 29.4 / 1.4 | 1.4 / 30.5 | **0.044** |
| infershard_qwen1_5b_tp4pp4, node 0 / node 1 | 8.2 / 1.5 | 1.5 / 8.3 | **0.18** |
| replicated inference, s1/s8 anything | ~0 | ~0 | noise |

1. **Symmetry separates every cross-node training cell from every cross-node
   serving cell in this data.** Pipeline-parallel serving sends activations
   forward only, so each end of a two-stage pipeline is one-directional.
   Pipeline-parallel *training* sends activation gradients back, so it is
   symmetric. Two serving cells is a small sample; tensor-parallel and MoE
   expert-parallel serving across nodes, which would be symmetric, have not been
   collected.
2. **The ceiling is 0.956, not 1.0, because of how the kernel counts.** Received
   bytes include TCP/IP headers (~4.6% at a 1460-byte MTU); transmitted bytes
   are counted before segmentation and do not. Any symmetry threshold has to
   allow for it.
3. **Training nodes move in lockstep; serving nodes do not.** Correlation of
   node 0's and node 1's transmit rate, active samples only: FSDP 0.90–1.00,
   serving 0.07–0.24.
4. **At 1 Hz on this network FSDP shows no periodicity.** It saturates the link
   continuously (duty cycle 0.97, no FFT peak). DDP ResNet-50 shows a ~3 s
   oscillation. Periodicity as a feature is fabric-dependent: on a faster
   network FSDP would burst and idle; here it never idles.
5. **Volume separates default configurations by three orders of magnitude**
   (FSDP-30B 17,000 MiB/s against pipeline serving 29), and the branch shows
   the same thing is spoofable: Robi's MN-P attack (pipeline parallelism across
   nodes, FSDP within) cuts training's NIC traffic 44×.

**Consequences for this repo.**
- The generator models pipeline-parallel serving as symmetric at node level.
  Real vLLM pipeline serving is directional. The inference `p2p` flow needs to
  send forward only, which changes the detector population: of its five serving
  families, only the MoE one remains symmetric across nodes.
- Symmetry therefore clears pipeline-sharded and disaggregated serving. What it
  cannot clear is tensor-parallel or expert-parallel serving across nodes, and
  those are the collection gap that matters most for this repo.
- MN-P is pipeline *training*, so it should stay symmetric while its volume
  falls 44×. That is a testable prediction: the MN-P NIC files are in LFS.
- The collector should keep raw cumulative counters, sample at 10 Hz and save on
  SIGTERM. All three are small changes to `scripts/collect_host_telemetry.py`.

## 4. What Robi's analysis establishes

From `classifier/rebuttal_analysis/results/tpu_multislice_findings.md`
(§22–33 are the multi-node GPU sections) and the results files.

- **Family-level holdout, multi-node, three-way:** test accuracy 0.569 against a
  majority-class baseline of 0.476, skill +0.092 with 95% CI [−0.02, +0.21] —
  not significant. Validation–test correlation is zero. (§32)
- **Only two honest multi-node training programs exist** (FSDP and DDP
  ResNet-50). Leave-one-family-out detects 0.5% of FSDP and 4% of DDP.
- **Unseen training configurations are missed:** DDP ResNet-50 at batch 512
  (46/48 runs), FSDP-7B without checkpointing (42/48). (§29a)
- **Unseen inference false-alarms:** 72B long-context serving is called training
  48/48; the worst offender is a small model at high parallelism. Six of the
  top eight features are memory levels: vLLM preallocates the card, so "full
  card" has come to mean "inference". (§24–27, §33)
- **Training is bimodal by topology:** multi-node training is quiet per GPU
  (188 W, 4–7% memory activity), single-node training busy. (§26–27)
- **Run-level results depend on the aggregation rule** — the same cell reads as
  missed 87.5%, caught 100% or missed 100%. (§30)
- **Evasion:** MN-I with a saturating filler evades at training duty ≤ 0.2;
  MN-O (classifier-gated) never trains once the detector is warm; MN-P cuts NIC
  traffic 44× but is still caught per GPU.

**Overlap with this repo.** Robi's results independently show the coverage
failure (unseen configurations and families missed, fold-dependent scores) and
the collapse under family holdout. This repo adds what his does not have: the
mechanism level of the grouping ladder, the control showing the collapse is not
a data-volume effect, the transfer matrix, the family budget, and the whole
interconnect analysis — symmetry, timing, the padding attack, the sampling spec.
None of that duplicates his work, and his NIC data is the first real test of it.

## 5. The run queue, for reuse

`scripts/rebuttal_collection/multinode_queue.sh` defines ~169 cells per
repetition across s16, s8 and s1: 600 s of telemetry each, three reps, about
11–13 minutes and $14–30 per cell on the H100 spot pair, so a full repetition is
~30 hours and ~$2.5k. A 21-cell workload-diversity round (LoRA, CLIP
contrastive, diffusion UNet, honest GPipe, DLRM, sharded FFT, stencil, NCCL
sweep, non-LLM inference) was smoke-tested on 09-23 and not yet collected.

**Defects to fix before the next collection:**
- `s16_infermem_qwen72b_u45` is defined twice with the same run id, so the
  second is always skipped.
- `s16_train_ddpstagger_k` and `s16_train_ddpkaccum8` are evasion strategies but
  are labelled honest training. `ddpstagger_k` also records no NIC traffic.
- FSDP-30B/70B default to SGD, so the "cannot fit on one node" 70B cell may fit.
- `s16_inferhf_shard72b*` is launched with 8 processes per node by a script that
  ignores the local rank, and is single-node despite its name.
- Replicated cells have no cross-node traffic whatever their scope label.
- Three nodes are not supported: the barrier, Ray shapes and MN-P's layer split
  assume two.
- Training models are random-initialised on synthetic tokens.

**Communication patterns with no cell:** tensor-parallel training; HSDP / ZeRO
hybrid (shard within a node, all-reduce across nodes); MoE expert-parallel
training or serving; disaggregated prefill/decode serving; tensor-parallel
serving across nodes; DiLoCo or local SGD on GPU; multi-node RL with rollout
workers and weight broadcast; a gradient-accumulation sweep; gradient
compression; and any NIC shaping or padding attack.
