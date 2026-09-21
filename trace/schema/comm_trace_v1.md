# Communication trace schema, v1

**What a per-node collection agent records, and what the detector is allowed to see.**
Parsa Salimi, SPAR F26. Draft for review — nothing here is frozen yet.

This document exists so that the detector can be built and tested before the
hardware collection works, and so that real traces drop into the same pipeline
without rework. It is deliberately conservative: the agent records what it read,
and all interpretation happens later.

---

## 1. The three rules

**1. Transmit and receive are never summed.** The schema has no field for total
bytes. This is not a stylistic preference: on an analytic sweep of 107,128
configurations, a threshold on total interconnect volume separates training from
co-located serving at AUC 0.51 — a coin flip — while adding tx/rx symmetry takes
a held-out-model classifier from 0.76 to 0.97. Transmit and receive are separate
hardware registers, so keeping them apart costs nothing. Summing them discards
the only feature that survives, and does so silently.

**2. The agent records raw counter values, never deltas.** Wraparound handling
and unit conversion are easy to get wrong and impossible to undo. A recorded raw
value can be reinterpreted years later; a recorded delta cannot. The reader does
the arithmetic.

**3. Nothing the agent records may depend on the workload's cooperation.** Byte
counters exposed by the driver are observable by the infrastructure owner.
Numbers obtained by hooking into the training process — NCCL callbacks, framework
instrumentation — are the adversary reporting on itself, and are excluded from
the observation table by construction. They may be recorded separately as ground
truth for calibration, clearly marked, and must never enter the feature set.

---

## 2. What a trace is

A trace is a directory:

```
<trace_id>/
  manifest.json       one object: schema version, timing, and (labelled data only) ground truth
  links.json          one entry per observed link: what it is, what units it reports
  observations.parquet   the samples
```

The label lives in `manifest.json` and never in `observations.parquet`, so it is
structurally impossible to train on it by accident.

### 2.1 `observations.parquet`

One row per (sample instant, link). Transmit and receive are columns of the same
row because they are read together from the same link at the same instant, and
because a row with both makes symmetry a single-row computation.

| column | type | meaning |
|---|---|---|
| `t_mono_ns` | int64 | agent's **monotonic** clock at read. Use for intervals: immune to NTP steps. |
| `t_wall_ns` | int64 | agent's **wall** clock at read. Use for cross-node alignment only. |
| `link_key` | string | foreign key into `links.json` |
| `tx_raw` | uint64 | transmit counter **exactly as read**, no conversion |
| `rx_raw` | uint64 | receive counter exactly as read |
| `read_dur_ns` | int32 | how long the read itself took; a long read means tx and rx are skewed relative to each other |
| `flags` | uint8 | bit 0: read error; bit 1: counter reset suspected; bit 2: source restarted |

`read_dur_ns` matters more than it looks. If reading tx and rx takes a
millisecond, the two counters are sampled a millisecond apart, which puts a floor
on how precisely symmetry can be measured on bursty traffic. Recording it lets us
find that floor instead of guessing.

### 2.2 `links.json`

Metadata that would otherwise be repeated on every row.

| field | meaning |
|---|---|
| `link_key` | stable identifier, e.g. `n03/gpu2/nvlink` or `n03/mlx5_0:1/ib` |
| `node_id` | which machine |
| `dev_id` | GPU index, NIC name |
| `link_id` | index within the device; `-1` if the counter is already an aggregate |
| `link_type` | `nvlink` \| `ib` \| `eth` \| `pcie` |
| `source` | provenance, e.g. `dcgm:1011`, `sysfs:port_xmit_data`, `sysfs:tx_bytes` |
| `unit` | `bytes` \| `words4` \| `flits` — **the source's unit, not ours** |
| `width_bits` | 32 or 64 — required to un-wrap correctly |
| `peak_bw_bytes_per_s` | nominal per-direction line rate, for utilisation and for the wrap check in §4 |

### 2.3 `manifest.json`

```jsonc
{
  "schema_version": "comm_trace/1",
  "trace_id": "...",
  "agent_version": "...",
  "sample_hz_nominal": 10.0,
  "nodes": {
    "n03": { "clock_offset_ns": 412000, "offset_method": "ntp|ptp|burst-correlation",
             "offset_uncertainty_ns": 250000 }
  },
  "ground_truth": {                    // OMIT ENTIRELY for traces to be scored
    "label": "training|inference|other",
    "workload": { "...": "free-form config" },
    "true_bytes": { "...": "optional, from an in-process hook; calibration only" }
  }
}
```

---

## 3. Units, widths and the conversions that bite

| source | unit | width | conversion to bytes |
|---|---|---|---|
| `/sys/class/infiniband/<dev>/ports/<n>/counters/port_xmit_data` | **4-byte words** | **32** | `× 4` |
| the same under `hw_counters/` or the `_64` variants | 4-byte words | 64 | `× 4` |
| `/sys/class/net/<if>/statistics/tx_bytes` | bytes | 64 | none |
| DCGM NVLink fields (1011 / 1012) | **verify on hardware** | verify | verify |

The ×4 on InfiniBand is the classic mistake — forget it and every inter-node
number is 4× low.

**The DCGM row is deliberately blank.** Those fields have never returned a real
value in our repo: of 2,050 telemetry files, 76 carry the NVLink columns and two
have a value, which is 2⁶³−1, NVIDIA's "not supported" sentinel. Whether they
report cumulative bytes or a rate, and in what unit, has to be established on
hardware before it is written down here. The schema carries `unit` and
`width_bits` per link precisely so this can be filled in without changing the
format.

---

## 4. Counter wraparound sets a floor on the sample rate

A 32-bit counter counting 4-byte words rolls over after 2³² × 4 = 17.2 GB. At
line rate that is fast:

| link | per-direction rate | 32-bit word counter wraps after | minimum safe sample rate |
|---|---:|---:|---:|
| IB NDR 400 Gb/s | 50 GB/s | **0.34 s** | **> 2.9 Hz** |
| IB HDR 200 Gb/s | 25 GB/s | 0.69 s | > 1.5 Hz |
| 100 GbE | 12.5 GB/s | 1.37 s | > 0.7 Hz |
| NVLink 4, per GPU | 450 GB/s | 0.04 s | > 26 Hz |

64-bit counters wrap after decades and can be ignored.

**Consequence: at the 1 Hz used by the existing collector, a 32-bit InfiniBand
counter on an NDR link wraps about three times between samples, and every one of
those wraps is invisible.** The delta would be wrong, not missing — which is the
dangerous kind.

Two requirements follow:

- **Prefer the 64-bit extended counters wherever they exist.** Record
  `width_bits` so the reader knows which it got.
- **Where only 32-bit counters are available**, sample at least 4× the wrap rate
  and treat any interval where the implied rate exceeds `peak_bw_bytes_per_s` as
  evidence of a missed wrap (set `flags` bit 1).

The reader un-wraps as `delta = (v2 − v1) mod 2^width_bits`, which is correct for
at most one wrap per interval and silently wrong beyond that. There is no way to
detect a double wrap from the counter alone — only from knowing the line rate.
Hence `peak_bw_bytes_per_s` in `links.json`.

---

## 5. Clocks

Each node timestamps with its own clock, so two things are needed.

**Within a node**, use `t_mono_ns`. A monotonic clock cannot jump backwards when
NTP corrects, which a wall clock can, and a backwards jump produces a negative
interval and a nonsensical rate.

**Across nodes**, use `t_wall_ns` plus the per-node `clock_offset_ns` in the
manifest. NTP on a LAN is typically within a millisecond; PTP is within
microseconds. For rate and symmetry features, millisecond skew is irrelevant. It
matters only for asking whether two nodes burst *simultaneously*.

Which leads to something worth exploiting rather than merely tolerating: **an
all-reduce makes every participating node transmit at the same instant.** So
cross-correlating nodes' transmit rate series recovers the clock offset — and the
existence of a sharp correlation peak is itself evidence of collective
communication, independent of volume. Inference serving has no reason to make
eight nodes burst in lockstep. This is a detection idea that falls out of the
clock-alignment problem, and the schema supports it by recording `t_wall_ns`
alongside the monotonic clock. `offset_method: "burst-correlation"` is reserved
for an offset estimated this way.

---

## 6. What the detector may use

Derived by the reader, from the columns above and nothing else:

| feature | from |
|---|---|
| tx rate, rx rate, per link and aggregated per fabric | counter deltas ÷ monotonic interval |
| **symmetry** = min(tx,rx) / max(tx,rx) | the two columns of one row |
| burstiness: CV of the rate series, autocorrelation, dominant FFT period | the rate series |
| cross-node correlation of transmit bursts | aligned `t_wall_ns` |
| link utilisation | rate ÷ `peak_bw_bytes_per_s` |

Explicitly **not** available, and not to be smuggled in:

- individual message sizes
- the number or type of collectives
- anything from a framework hook

Message size is the strongest single feature in the analytic model (AUC 0.99) and
is unobtainable from a cumulative counter at any sample rate — a counter says how
many bytes crossed, never in what sized pieces. A result that depends on it is a
result that depends on the adversary's cooperation.

---

## 7. Open questions for the hardware side

1. Do DCGM fields 1011/1012 return cumulative bytes or a rate, in what unit, and
   at what width? They have never produced a real value in our data.
2. Are the 64-bit extended InfiniBand counters exposed on the cluster's HCAs?
3. Are NVLink counters per-link or already aggregated per GPU?
4. What does a `read_dur_ns` distribution look like in practice — is the tx/rx
   read skew small enough to measure symmetry on millisecond-scale bursts?
5. What is the actual fabric between the two current nodes: InfiniBand or
   Ethernet? This changes `peak_bw_bytes_per_s` by 4× and the wrap floor with it.
