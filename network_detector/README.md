# Network-only detection on real multi-node data — results

**Can a provider tell training from everything else using only each node's NIC
byte counters, on jobs from workload families it has never seen?** Tested on
Robi Rahman's multi-node corpus (`multinode-tpu-experiments` branch): 2 × 8 H100
or A100, one NIC per node, per-GPU telemetry and per-node NIC logs recorded
together. Method in [`METHOD.md`](METHOD.md).

---

## 1. The result

Honest two-node jobs, every family held out in turn, identical windows for every
detector (7 families, 63 jobs, 2,554 windows):

| detector | catches training | false alarms (windows) | false alarms (jobs) |
|---|---:|---:|---:|
| learned per-GPU model — Robi's 174 NVML/DCGM features | 5% | 70% | 68% |
| learned network model | 74% | 72% | 77% |
| learned model on both | 17% | 82% | 80% |
| **network rule: traffic present and balanced** | **92%** | **0.2%** | **0%** |
| network rule, falling back to per-GPU when the network is quiet | 92% | 11.5% | 2.3% |

By held-out family, share of windows called training:

| family | learned per-GPU | network rule |
|---|---:|---:|
| **training** — DDP ResNet-50 | 13% | **81%** |
| **training** — FSDP 1B / 7B / 30B | 0% | **99.8%** |
| **serving** — vLLM pipeline-sharded, 12 configurations incl. 72B at 32k context | **83%** | **0%** |
| **serving** — replicated, no cross-node traffic | 0% | 0% |
| **non-ML** — all-reduce loop (`nonml_comm`) | 98% | 7% |
| **non-ML** — compute-bound HPC, saturation | 0% | 0% |

**What it says.**

- **A learned detector cannot generalise across workload families here, on
  either channel.** With a family held out, the training set has one training
  program left and no example of the held-out mechanism. The per-GPU model
  catches 5% of unseen training while false-alarming on 83% of unseen
  pipeline-sharded serving, reproducing Robi's §25/§32. The learned network
  model fails the same way for the same reason: with pipeline serving held out,
  it has never seen one-directional traffic labelled "not training", so it
  learns "traffic present means training".
- **A rule derived from the mechanism does not need coverage.** Gradient
  averaging is an all-reduce, and an all-reduce sends as much as it receives
  whatever the model or framework. On real hardware every cross-node training
  job reads 0.954–0.961 per-node symmetry and every pipeline-serving job
  0.04–0.18, so the rule is insensitive to where the thresholds sit: 92.2–92.4%
  caught, 0.2–1.4% false alarms, over 30 combinations spanning symmetry
  0.5–0.95 and activity 0.1–10 MiB/s.
- **The two channels see different things, and the network is the one that
  generalises on this corpus.** Falling back to the per-GPU model when the
  network is quiet adds its false alarms (pipeline serving's quiet windows,
  the all-reduce loop) without adding catches.

## 2. Adversarial cells

The rule is fitted to nothing, so it has not seen these either. Share of windows
called training:

| attack | learned per-GPU (all honest families) | network rule |
|---|---:|---:|
| MN-I interleave, training duty 2 / 5 / 10–50% | 100 / 100 / 100% | 64 / 95 / 100% |
| MN-I with saturating filler, duty 5 / 20 / 50% | 27 / 92 / 100% | 42 / 92 / 100% |
| DDP with 8-step gradient accumulation | 95% | 95% |
| MN-O oracle-gated, thresholds 0.5–0.9 (never trains) | 0% | 0% |
| filler / saturating filler only (never trains) | 97% / 0% | 0% / 0% |
| DDP "stagger" (trains, but no cross-node traffic) | 96% | **0%** |

- **Interleaving does not hide training from the network.** Whenever it trains,
  it synchronises, and synchronisation is loud and balanced. The rule catches it
  at the rate it actually trains.
- **The network is silent on cells that never train**, which is the correct
  answer. The per-GPU model calls the non-saturating filler-only cell training
  97% of the time.
- **The blind spot is training without cross-node synchronisation.** The
  `ddpstagger_k` cells keep 16 GPUs busy and send nothing between nodes: to the
  network they are two independent single-node jobs. DiLoCo-style training that
  synchronises rarely will look the same between syncs. This is the network
  channel's structural limit, and the case where per-GPU telemetry has to carry
  the decision.

## 3. What this does not establish

- **The thresholds were set after seeing this corpus.** The rule's form comes
  from the comm model; its two numbers were chosen with the per-cell medians in
  view. The insensitivity above makes this matter little here, but the rule's
  held-out test is the next collection round. It is frozen, with predictions, in
  [`PREDICTIONS.md`](PREDICTIONS.md).
- **Seven honest families, two of them training.** Every "unseen family" number
  rests on one or two families.
- **The comparison is between a rule and learned models.** It shows that the
  mechanism generalises where learning from seven families cannot, not that
  network telemetry beats NVML in general. A per-GPU model trained on the
  families it will meet is near-perfect in distribution.
- **One network, and not a production one.** One TCP gVNIC per node, NCCL over
  sockets. The 0.956 ceiling is a TCP header artifact; an RDMA fabric reads
  closer to 1.0.
- **The rule has not faced an attack aimed at it.** Every attack in the corpus
  targets the per-GPU detector. What evading the network would cost is worked
  out in [`ATTACKS.md`](ATTACKS.md): rare synchronisation is the structural
  limit, and a lopsided parameter server is free on two nodes but a real tax at
  scale.
- **Symmetric non-ML communication is untested at scale.** The only example is
  one `nonml_comm` job whose loop is mostly idle. A busier all-reduce, an FFT
  all-to-all, a stencil halo exchange, DLRM's all-to-all, or tensor- and
  expert-parallel serving across nodes should all read as training. The next
  collection round contains four of these.

## 4. Corpus issues found along the way

- **Six honest "training" jobs are idle hardware**: GPUs at 0% utilisation with
  0.5 GB loaded, labelled `s16_train_ddp_resnet50` (2), `s16_train_fsdp7b_seq2k`
  (2), `s16_train_fsdp30b` (1) and `s16_train_fsdp7b` (1). Listed in
  `results/idle_jobs_dropped.csv`.
- **`s16_train_ddpstagger_k` has no cross-node traffic** while its GPUs are busy,
  so it is not the DDP variant its name describes.
