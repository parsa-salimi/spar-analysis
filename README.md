# spar-analysis

Analysis work for the SPAR Fall 2026 project on **detecting distributed model
training from interconnect telemetry**.

Parsa Salimi. Mentor: Will Fowler. Co-mentee: Long Yi.

**Start with [`STATUS.md`](STATUS.md)** — what exists, what it found, what is
blocked on whom. [`RELATION_TO_PRIOR_WORK.md`](RELATION_TO_PRIOR_WORK.md) sets
out how these numbers relate to the paper's and to earlier analysis of the
corpus; read it before quoting one against the other. — what exists, what it found, what is
blocked on whom. Two minutes. [`GLOSSARY.md`](GLOSSARY.md) defines every term
either file assumes.

This is my working repository, not the team's record. The shared repo is
`williamfowler/GPU-monitoring`; this one holds analysis that reads from it and
is kept separate so it can move fast without touching a repo three people
commit to.

---

## What the project is

A cloud provider can already see, for every GPU it rents out, how much data
crosses the interconnect — NVLink inside a server, InfiniBand or Ethernet
between servers. The question is whether that alone reveals when someone is
training a large model, as opposed to serving one.

That matters because compute-governance proposals lean on training runs being
*detectable*. If they are, reporting thresholds can be checked. If they aren't,
or if they are cheap to evade, those proposals need rewriting.

Prior work (Rahman & Tajdari, arXiv 2606.19262) does this with nine **on-GPU**
counters on **single nodes**, reporting 98.2% in-distribution accuracy and
43–87% against unexpected adversarial workloads, and names multi-node
interconnect as its first stated limitation. Our project changes the
signal to **interconnect bytes** and the scale to **multi-node**, because the
threat we care about — Seferis & Fist's "compute structuring" — is an adversary
splitting a run so no single node looks like training.

---

## What's here

Two pieces of work, each self-contained with its own write-up.

### `power_analysis/` — how much can this corpus actually support?

Accuracy depends strongly on what is held out. The paper's protocol groups by
run, and the row below marked as such reproduces its result. The lower rows ask a
stricter question — generalisation to a kind of workload never seen before.

| held out | groups | accuracy |
|---|---:|---:|
| nothing (random windows) | — | 0.991 |
| run (one GPU stream) — *the prior protocol* | 1,000 | 0.977 |
| job (one physical execution) | 651 | 0.966 |
| workload label | 144 | 0.874 |
| **workload family** | **75** | **0.724** |
| mechanism class | 22 | 0.523 |

Leave-one-family-out gives **0.715 ± 0.080**; leave-one-*mechanism*-out gives
**0.520 ± 0.150**, below the 0.68 you get by always guessing "training".

Three consequences: validation cannot rank models here (ρ = −0.14; picking the
best of ten splits by validation does *worse* than picking blind); the wide
split-to-split range is sampling noise, not model behaviour; and precision is
bought with **families, not GPU-hours** — a 7.2× increase in windows leaves the
spread unchanged. Reaching ±0.10 needs ~113 families against the 75 we have.

Read `power_analysis/METHOD.md` first — it walks through the whole thing
assuming no background.

### `comm_model/` — is there a signal at all?

The project plan puts an early-stopping gate at week 4. Much of it is closed-form
arithmetic, so it can be answered without hardware. Over 107,128 realistic
configurations:

- **A volume threshold is at chance.** AUC 0.51 against co-located serving, 0.59
  against disaggregated serving. 100% of training configurations lie inside
  inference's 5–95% band. Volume alone doesn't even have a stable *direction*
  across model scales.
- **tx/rx symmetry is the feature that works**: held-out-model AUC 0.757 → 0.965.
  Gradient synchronisation sends what it receives; a KV-cache push doesn't. And
  transmit and receive are separate hardware registers, so it is free to collect.
- **Message size scores higher still — and is invisible to a byte counter.**
  Getting it needs an in-process hook, i.e. the adversary reporting on itself, so
  it must stay out of the feature set.
- **Three of four evasions are free.** DiLoCo cuts inter-node traffic 1,500× at
  no wall-clock cost, and *gains* 2.89× throughput when the baseline was
  network-bound. Its convergence penalty is not computable here and is left
  deliberately unmodelled.

This settles the collection-agent design: record **tx and rx separately, per
link**. A schema that sums them throws away the only feature that works.
`comm_model/METHOD.md` walks through how this was computed, including the traps
this kind of model has to be built around.

### `trace/` — the format the collector will write

Defines the trace schema and generates synthetic traces in it, so the detector
can be built and tested before the hardware collects anything. Found along the
way that a 32-bit InfiniBand counter wraps in 0.34 s at line rate — the existing
collector samples at 1 Hz — and that the obvious guard against that does not
work. See `trace/README.md`.

---

## Running it

```bash
pip install -r requirements.txt
```

`comm_model/` is self-contained — start with the anchor tests, which pin the
arithmetic to numbers derivable by hand:

```bash
cd comm_model
python3 test_commvol.py      # 20 anchors; if these fail, nothing else means anything
python3 sweep.py             # rebuild the configuration grid and separability numbers
python3 evasions.py          # evasion cost curves
python3 figures_comm.py      # figures
```

`power_analysis/` needs the team repo for its feature extractor and label
taxonomy. Clone it next to this one, or set `GPU_MONITORING_ROOT`:

```bash
git lfs install                                                # REQUIRED
git clone https://github.com/williamfowler/GPU-monitoring.git  # private, ~2.8 GB
```

GitHub's web-UI "Download ZIP" does **not** fetch LFS objects — you get 130-byte
pointer files. Then:

```bash
cd power_analysis
python3 analyse.py           # summary statistics from the committed results
python3 figures.py           # the four figures
python3 sensitivity.py 10 40 # window-cap sensitivity check
```

The windowed feature cache (`data/windows_dc3.parquet`, 12 MB) is committed, so
the analysis re-runs without touching the 2.8 GB telemetry corpus. Regenerating
it from scratch needs the corpus:

```bash
python3 inventory.py         # corpus census
python3 extract.py 0 350     # windows + features, in chunks
python3 power.py family 40 1000
python3 lofo.py family 0 75
```

---

## Notes for anyone reading the code

- **`fastwin.py` reproduces the team repo's causal windowing bit-for-bit**
  (verified: max absolute difference 0.0 across all numeric feature columns) but
  in O(n) rather than O(n²). The original re-slices the whole run with a boolean
  mask for every window, which is why the eight-hour runs are slow. Worth
  upstreaming.
- **Never cast the window timestamp columns to float32.** Epoch seconds are
  ~1.79 × 10⁹, where float32 has about 128 seconds of resolution; the cast
  silently collapses distinct windows onto the same timestamp; a later
  deduplication step then discards most of the corpus, with no error raised.
- **The ring all-reduce factor is 2(N−1)/N, not 2.** At world_size 2 it is
  exactly 1.0, so the common shorthand over-counts two-GPU runs by 2×.
- **`families.py` encodes a judgement call.** The full 179-label mapping is
  dumped to `power_analysis/results/family_map.csv` so it can be argued with, and
  three levels of coarseness are reported so the conclusion can be checked
  against where the line is drawn.

## Caveat that governs the comm model

Its numbers come from an analytic grid I designed, not from measurements. Every
AUC there is a feasibility bound under a stated configuration prior — it answers
"could a perfect observer tell these apart?", not "what will our detector score?"
Given that the power analysis watched 0.99 become 0.52 once the evaluation
respected the real unit of replication, these figures are for deciding what to
build and what to collect, not for reporting as detection performance.
