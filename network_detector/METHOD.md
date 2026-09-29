# Network-only detection — how it works

**Written assuming no background.** Results in [`README.md`](README.md); see
[`../GLOSSARY.md`](../GLOSSARY.md) for terms.

## 0. The question and why it matters

Robi Rahman's detector reads per-GPU telemetry (power, memory, utilisation) from
NVML, which runs on the customer's own machine. A customer who controls the
machine can tamper with it. A provider can count the bytes each node sends and
receives at the network, outside the customer's reach. If training can be told
apart from everything else from those counters alone, the detector is one a
provider can run against an adversarial customer.

## 1. The data

Robi's multi-node corpus: pairs of 8-GPU nodes on Google Cloud (H100 or A100),
one network card each, NCCL over TCP. Every job records per-GPU telemetry and,
per node, a once-a-second log of bytes sent (tx) and received (rx). Only jobs
spanning both nodes (`s16_`) are used: a single-node job puts nothing on the
network, so the network has nothing to say about it.

Files are read straight from the branch without checking it out
(`data.py`). A job is one run id; the two channels are joined by node and time,
since that is what a provider observes (`build.py`).

## 2. What is computed

Both channels use 30 s windows at a 15 s stride.

**Per-GPU** (`gpufeat.py`): Robi's own feature code, unchanged — 174 statistics
over nine NVML signals. Four GPUs per job (0 and 4 on each node); the job's score
in a window is the median over them.

**Network** (`netfeat.py`), per node, averaged over the nodes:

| feature | meaning |
|---|---|
| `vol` | log of the mean byte rate |
| `sym` | min(sent, received) / max(sent, received) |
| `dir_abs` | how one-directional the node is |
| `duty` | share of seconds with traffic |
| `cv`, `acf1`, `acf3`, `fft_peak_frac`, `peak_mean` | how uneven, smooth, periodic and spiky the rate is |
| `burst_sym` | symmetry of the part above the window's floor |

## 3. The detectors

- **Learned per-GPU, learned network, learned on both**: a 200-tree random
  forest, Robi's best-performing configuration, no tuning.
- **The network rule** (`rule.py`): a window is *active* if its mean rate is at
  least 1 MiB/s, and is called *training* if it is active and its symmetry is at
  least 0.8. Nothing is fitted. The reasoning is mechanical: gradient averaging
  is an all-reduce, an all-reduce sends what it receives, so training that
  synchronises across nodes is balanced; pipeline serving only sends forward.
- **Combined**: the rule when the network is active, the learned per-GPU model
  when it is not.

## 4. How they are tested

- **Leave one family out.** A family is a workload with its size and parameter
  sweeps collapsed (Robi's definition). Each honest family is held out in turn;
  the learned models train on the rest.
- **Idle screen.** Honest jobs in which every sampled GPU sits below 5%
  utilisation with under 2 GB loaded are dropped: they carry a training label but
  nothing ran.
- **Same windows everywhere.** Only windows present in both channels are scored.
- **Reported:** share of each held-out family's windows called training, and at
  job level the share of jobs whose majority of windows are.
- **Adversarial jobs** are scored by models fitted on every honest family, and
  never trained on.

## 5. Reproducing

```bash
# once: in the GPU-monitoring clone, fetch the branch's LFS objects
git lfs fetch origin origin/multinode-tpu-experiments --include="data/gcp/**,gcs_c_quarantine/**"

python3 -m network_detector.data        # index the branch's files
python3 -m network_detector.build       # windows for both channels -> data/
python3 -m network_detector.evaluate    # every table in README.md -> results/
```

`GPUMON_REPO` points at the clone if it is not at `~/mnt/Bertie/GPU-monitoring-new`.
