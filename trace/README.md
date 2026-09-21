# Trace schema and synthetic agent

**How a detector gets built before the hardware can collect anything.**
Parsa Salimi, SPAR F26. September 2026.

See `GLOSSARY.md` at the repo root for any unfamiliar term.

---

## Why this exists

The detector reads traces. Nothing produces traces yet: in the whole team repo,
the NVLink counters have returned a real value **zero times** — of 2,050
telemetry files, 76 carry the NVLink columns and two hold a value, which is
2⁶³−1, NVIDIA's "not supported" sentinel.

Waiting is the obvious plan and a bad one. If the hardware slips past
November 1, we have nothing for the midterm, and every downstream piece —
aggregator, features, classifier, evaluation — sits idle in the meantime.

So instead: **define the file format the collector will write, and write a fake
collector that produces files in that format** from the analytic model in
`comm_model/`. Everything downstream reads the format rather than the hardware,
so real traces drop in unchanged.

There is a second reason, which I think is the better one. **On synthetic traces
the true byte counts are known exactly**, so the accuracy lost to the *measuring
apparatus* — sampling too slowly, clocks disagreeing between machines, counter
quantisation — can be separated from the accuracy lost to an adversary hiding.
On real hardware those two are hopelessly confounded. This is how we find out
what sample rate the agent needs *before* Long builds it, rather than guessing.

---

## The three rules

Each is enforced by the schema rather than by discipline.

**1. Transmit and receive are never summed.** The observation table has no field
for a total, so the mistake is unrepresentable. This is not taste: on the
analytic sweep, a threshold on total volume separates training from co-located
serving at AUC 0.51 — a coin flip — while adding tx/rx symmetry takes a
held-out-model classifier from 0.76 to 0.97. Transmit and receive are separate
hardware registers, so keeping them apart is free. Summing them discards the
only feature that survives, silently.

**2. The agent records raw counter values, never deltas.** Unwrapping and unit
conversion are easy to get wrong and impossible to undo. A raw value can be
reinterpreted later; a delta cannot. The reader does the arithmetic.

**3. The label lives in the manifest, never in the observation table** — so it
cannot leak into a feature matrix by accident.

---

## What a trace is

```
<trace_id>/
  manifest.json          schema version, sample rate, clock offsets, and
                         (labelled data only) the ground truth
  links.json             one entry per link: what it is, what units it reports,
                         how wide its counter is, its line rate
  observations.parquet   one row per (sample instant, link):
                         t_mono_ns, t_wall_ns, link_key, tx_raw, rx_raw,
                         read_dur_ns, flags
```

Two clocks per sample, deliberately. `t_mono_ns` is monotonic and used for
intervals — it cannot jump backwards when NTP corrects, which a wall clock can,
and a backwards jump produces a negative interval and a nonsense rate.
`t_wall_ns` is only for lining nodes up with each other.

`read_dur_ns` matters more than it looks: if reading tx and rx takes a
millisecond, the two counters are sampled a millisecond apart, which puts a floor
on how precisely symmetry can be measured on bursty traffic. Recording it lets us
find that floor instead of assuming it away.

The full specification is `schema/comm_trace_v1.md`.

---

## The finding that came out of building it

Network counters fill up and roll back to zero, like a car odometer. On
InfiniBand the standard sysfs counter holds 32 bits of **4-byte words** — about
17.2 GB — which at NDR line rate fills in **0.34 seconds**.

| link | per-direction rate | 32-bit word counter wraps after |
|---|---:|---:|
| IB NDR 400 Gb/s | 50 GB/s | **0.34 s** |
| IB HDR 200 Gb/s | 25 GB/s | 0.69 s |
| 100 GbE | 12.5 GB/s | 1.37 s |
| NVLink 4, per GPU | 450 GB/s | 0.04 s |

**The existing collector samples at 1 Hz.** On an NDR link that is roughly three
roll-overs between readings, and the reading you get is wrong rather than
missing — which is the dangerous kind.

### And the obvious guard against it does not work

I wrote the natural check: if the implied rate exceeds the link's physical
maximum, a roll-over was missed. The synthetic traces proved it useless. Past
**one** roll-over the modular delta is an arbitrary value in [0, 17.2 GB), which
divided by the sample interval is comfortably *below* line rate — so nothing
trips. In test, the reader recovered **13.9% of the true bytes and raised no
alarm.**

```
PASS  the RATE heuristic is blind to multi-wrap, as predicted
      [0% flagged by rate, though the counter wraps every 5 ms
       and is sampled every 100 ms]
PASS  the CONFIG check catches it -- sample interval exceeds the wrap period
PASS  preflight refuses the link before a single value is read
      [min safe rate 838 Hz vs 10 Hz configured]
```

The real guard is a **configuration** check made before collection starts,
comparing the sample interval to the counter's roll-over period — decidable from
`links.json` alone, without looking at a single value. That is `preflight()`.

**Practical consequence:** use the 64-bit extended counters wherever they exist.
The standard 32-bit sysfs IB counters are unsafe below ~93 Hz on an 8-NIC node.
This is now an open question for the hardware side (see `STATUS.md`).

---

## What's here

```
schema/comm_trace_v1.md   the specification -- the document Long needs
commtrace/schema.py       record types, on-disk layout, validation, wrap arithmetic
commtrace/gen.py          synthetic generator, built on comm_model/commvol.py
commtrace/reader.py       unwrap, convert, rate series, features, preflight()
test_trace.py             13 round-trip and safety checks
```

```bash
cd trace && python3 test_trace.py
```

The tests are the point. They assert that recovered bytes match the generator's
ground truth to within 0.2%; that the wrap guard fails exactly where predicted
and the configuration check catches it; that symmetry separates training from
disaggregated serving at node level **while the same serving trace stays
symmetric on NVLink** (so symmetry is not a class label in disguise); and that
the schema refuses a summed-bytes column.

---

## Two corrections the generator forced on the analytic model

**Symmetry is a per-node property, not a per-transfer one.** `commvol` attaches
`sym` to a flow, and one pipeline send really is strictly directional — but a
middle pipeline stage receives from its predecessor as much as it sends to its
successor, so at node level it is symmetric. The genuinely asymmetric case is
narrower than first claimed: the KV-cache push, where prefill nodes only send
and decode nodes only receive. The generator derives symmetry from accumulated
tx/rx rather than from the flow's label.

**Transfers are ramps, not impulses.** A 700 GB all-gather takes 1.75 s at line
rate; the counter rises steadily through it. Modelling transfers as instantaneous
produced sample intervals implying **2.9 TB/s on a 400 GB/s link** — physically
impossible, and it fired the wrap heuristic as a false positive. Caught by the
tests, fixed in `_cumulative_at`.

---

## The measurement-artifact study

Done — see [`METHOD.md`](METHOD.md). Sweeping sample rate, counter width, clock
skew and tx/rx read skew over 12 workload families gives the spec line:

> **Sample at ≥ 2 Hz, use the 64-bit counters, ordinary NTP is fine.**

Counter width matters more than sample rate. Clock skew destroys the cross-node
synchrony feature (0.94 → 0.03 from perfect sync to 1 s) without hurting
detection, because the other features carry the signal redundantly.

The study's resolution is one workload family = 0.083 AUC, so most knobs come
back "no measurable effect" rather than "no effect" — the same
precision-is-bought-with-families arithmetic as the power analysis.

## Next

Build the detector: aggregator, the full feature extractor, and the two-stage
classifier whose second stage is blind to absolute byte volume.
