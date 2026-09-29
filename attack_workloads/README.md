# Attack workloads aimed at the network detector

Two training programs that try to defeat `network_detector/rule.py` — a node is
called training when it moves traffic and sends about what it receives — each
exploiting a different half of it. Written to drop into Robi Rahman's
`multinode-tpu-experiments` workloads and run queue unchanged. Why these two, and
what they should cost, is in [`../network_detector/ATTACKS.md`](../network_detector/ATTACKS.md).

| script | attacks | how |
|---|---|---|
| `train_diloco.py` | "moves traffic" | DiLoCo: each node trains alone for H steps, then the nodes average. Silent between syncs; each sync a burst or, with `--schedule spread`, a trickle over the next H steps. Optional int8/int4 outer gradient. |
| `train_ps_lopsided.py` | "sends what it receives" | Parameter server on node 0; workers upload top-*c* gradient entries with error feedback and download dense weights. Per-node symmetry ≈ 3c. |

Both use Robi's conventions: torchrun across two nodes of 8, his
`RandomTransformerLM` on synthetic tokens, `CollectiveDeadline` for a shared stop,
STEP lines from rank 0, and `advtrain` in every cell name so the label routing
and honest mask treat them as adversarial training. Cross-node traffic is
hierarchical: GPU *r* on each node handles 1/8 of the model, so a node moves one
model's worth per exchange, not eight. Each rank also writes a `NETLOG` line —
exactly what it put on the network, per step — so the NIC logs can be checked
against the intended shape.

## Install

Copy `_mn_attack.py`, `train_diloco.py` and `train_ps_lopsided.py` into
`workloads/`, and the lines in `queue_cells.sh` into the S16 phase of
`scripts/rebuttal_collection/multinode_queue.sh`. Both need a full 1B replica with
optimizer state per GPU, so 80 GB cards.

## Tested

`python3 test_cpu.py` runs every check on CPU with the gloo backend, 2–6
processes standing in for 2–3 nodes (`robi_stub/` holds copies of the three
helpers from Robi's branch that the scripts import). 22/22 pass:

- **Maths, exactly.** DiLoCo's outer Nesterov step and the parameter server at
  c = 1 match independent single-process reference implementations to the last
  bit (fp32 wire).
- **Agreement.** Every rank on every node ends with identical weights.
- **Signature.** DiLoCo burst: traffic only on sync steps, one bf16 model per node
  per sync, balanced. DiLoCo spread: one equal piece on every step. Int4: 0.54
  bytes/param. Parameter server at c = 0.1 over three nodes: every node reads
  0.300 symmetry, the server sending more than it receives and the workers the
  reverse.
- **Shutdown.** Timed runs end cleanly with every rank on the same step.

Not tested, because it needs GPUs: NCCL point-to-point across real nodes, memory
on a real card, and step times. Smoke-test one cell of each first (`SMOKE=1`).

## Predicted calls by the frozen network rule

Step times are estimates (a 1B DDP step at batch 4, sequence 1024 on H100 ≈
0.2 s); sync counts scale with the real ones. Symmetry on Robi's TCP NIC reads
about 0.04 higher than the ledger's, from acknowledgements.

| cell | cross-node traffic | predicted windows called training |
|---|---|---|
| `diloco_h100` | 2.6 GB burst every ~20 s, balanced | ~100% — caught |
| `diloco_h500` | 2.6 GB burst every ~100 s | ~30% — **missed by a majority-of-windows job call**, caught by an any-window call |
| `diloco_h100_int4` | 0.7 GB burst every ~20 s | ~100% — caught |
| `diloco_h100_spread` | ~130 MB/s continuous, balanced | ~100% — caught |
| `diloco_h500_int4_spread` | ~6 MiB/s continuous, balanced | ~100% — caught: for a 1B model in a 10-minute cell the trickle cannot get under 1 MiB/s |
| `pslop_c05` | every step, symmetry ≈ 0.19 | **~0% — evades** |
| `pslop_c10` | every step, symmetry ≈ 0.34 | **~0% — evades** |
| `pslop_c30` | every step, symmetry ≈ 0.94 | ~100% — caught (control) |

The lopsided parameter server is expected to defeat the rule outright on two
nodes — that is the point of running it: it measures the gap, and gives the next
defence (step cadence, per-GPU telemetry) something real to be tested on.

## Next attack to add

A `--pace` option for `train_ps_lopsided.py`: download the next step's
parameters in small, randomly timed pieces during the current step's compute
(one step of staleness). The cadence study (`../cadence/`) predicts it evades
burst-shape detection at every sample rate; this cell would test that on
hardware.
