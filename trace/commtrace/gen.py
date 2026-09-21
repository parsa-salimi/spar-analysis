"""Synthetic comm_trace generator.

Turns an analytic workload (commvol.Workload) into counter readings that look
like what a real agent would record: cumulative counters in the source's own
units, sampled at a finite rate with jitter, quantised, wrapped at the counter
width, and with tx and rx read a few microseconds apart.

Why bother, when we could wait for real traces: on synthetic traces the TRUE
byte totals are known exactly, so the loss caused by the measurement apparatus
(sample rate, clock skew, quantisation, read skew) can be separated from the
loss caused by the adversary. On real hardware those two are confounded and
cannot be separated at all.

A note on symmetry, because it corrects the analytic model. commvol attaches a
`sym` to each FLOW, which is a per-transfer property: one pipeline send is
strictly directional. But a monitor sees per-NODE aggregates, and a middle
pipeline stage receives from its predecessor as much as it sends to its
successor, so at the node level it is symmetric. This generator therefore
derives symmetry from actual per-node tx/rx accumulation rather than from the
flow's label. The classes still separate, but for a sharper reason: what is
genuinely asymmetric at node level is a KV-cache push, where prefill nodes only
send and decode nodes only receive.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import Link, Manifest, UNITS, write_trace

SYMMETRIC_KINDS = {"allreduce", "allgather", "reducescatter", "alltoall"}


@dataclass
class Collection:
    """How the agent collects, as opposed to what is happening on the wire."""
    sample_hz: float = 10.0
    sample_jitter_frac: float = 0.05     # sd of sample time, as a fraction of the period
    read_dur_ns: int = 80_000            # tx and rx are read this far apart
    clock_skew_sd_ns: float = 1e6        # per-node wall-clock offset, 1 ms default (NTP)
    ib_unit: str = "words4"
    ib_width: int = 32                   # the pessimistic default; 64 where available
    nvlink_unit: str = "bytes"
    nvlink_width: int = 64
    seed: int = 0


@dataclass
class Topology:
    n_nodes: int = 4
    gpus_per_node: int = 8
    roles: list[str] = field(default_factory=list)   # per node: train|prefill|decode|serve

    def resolved(self, cls: str) -> list[str]:
        if self.roles:
            return self.roles
        if cls == "training":
            return ["train"] * self.n_nodes
        half = max(1, self.n_nodes // 2)
        return ["prefill"] * half + ["decode"] * (self.n_nodes - half)


def _event_times(rate_per_s: float, duration: float, interval_cv: float,
                 rng: np.random.Generator) -> np.ndarray:
    """When transfers happen. interval_cv >= 0.9 is treated as a Poisson
    arrival process (request-driven); anything lower is a jittered periodic
    loop boundary."""
    if rate_per_s <= 0:
        return np.empty(0)
    n = int(rate_per_s * duration)
    if n < 1:
        return np.array([duration * rng.random()]) if rng.random() < rate_per_s*duration else np.empty(0)
    if interval_cv >= 0.9:
        gaps = rng.exponential(1.0 / rate_per_s, size=int(n * 1.4) + 8)
        t = np.cumsum(gaps)
        return t[t < duration]
    period = 1.0 / rate_per_s
    t = np.arange(n) * period + period * 0.5
    if interval_cv > 0:
        t = t + rng.normal(0, interval_cv * period, size=n)
    t.sort()
    return t[(t >= 0) & (t < duration)]


def _sizes(n: int, mean_bytes: float, size_cv: float, rng) -> np.ndarray:
    if n == 0:
        return np.empty(0)
    if size_cv <= 0:
        return np.full(n, mean_bytes)
    sigma = np.sqrt(np.log1p(size_cv ** 2))
    return rng.lognormal(np.log(mean_bytes) - sigma**2 / 2, sigma, size=n)


def _cumulative_at(sample_t: np.ndarray, ev_t: np.ndarray, ev_b: np.ndarray,
                   link_bps: float, duration: float) -> np.ndarray:
    """Cumulative bytes at each sample time.

    A transfer is a RAMP, not an impulse: moving B bytes over a link of
    capacity R takes B/R seconds, during which the counter rises steadily.
    Modelling transfers as instantaneous produces sample intervals with an
    implied rate far above line rate -- physically impossible, and it fires the
    reader's wrap heuristic as a false positive. (Found exactly that way: 0.67%
    of intervals reported 2.9 TB/s on a 400 GB/s link.)

    Built on a fine grid by adding +rate at each transfer's start bin and -rate
    at its end bin, then cumsum-ing twice: O(n_transfers + n_bins).
    """
    if ev_t.size == 0:
        return np.zeros_like(sample_t)
    dur = np.maximum(ev_b / max(link_bps, 1.0), 1e-6)
    fine = max(min(dur.min() / 4.0, 1e-3), duration / 4_000_000)
    nb = int(np.ceil(duration / fine)) + 2
    acc = np.zeros(nb + 2)
    r = ev_b / dur
    i0 = np.clip((ev_t / fine).astype(int), 0, nb)
    i1 = np.clip(((ev_t + dur) / fine).astype(int), 0, nb)
    np.add.at(acc, i0, r)
    np.add.at(acc, i1, -r)
    rate = np.cumsum(acc)[:nb]
    cum = np.cumsum(rate) * fine
    idx = np.clip((sample_t / fine).astype(int), 0, nb - 1)
    return cum[idx]


def synth_trace(workload, out_dir, *, duration_s: float = 300.0,
                topo: Topology | None = None, coll: Collection | None = None,
                trace_id: str | None = None, with_label: bool = True):
    """Generate one trace directory. Returns (path, truth) where truth carries
    the exact byte totals the generator laid down, for round-trip checking."""
    topo = topo or Topology()
    coll = coll or Collection()
    rng = np.random.default_rng(coll.seed)
    roles = topo.resolved(workload.cls)
    trace_id = trace_id or hashlib.sha1(
        f"{workload.label}|{duration_s}|{coll.seed}".encode()).hexdigest()[:12]

    links: list[Link] = []
    rows = []
    truth = {}

    for ni, (node, role) in enumerate(zip([f"n{i:02d}" for i in range(topo.n_nodes)], roles)):
        # ── lay down the true transfer events for this node ──────────────
        per_fabric = {"nvlink": {"tx": [[], []], "rx": [[], []]},
                      "ib":     {"tx": [[], []], "rx": [[], []]}}
        for fl in workload.flows:
            fabric = "nvlink" if fl.fabric == "intra" else "ib"
            node_rate = fl.bytes_per_s * topo.gpus_per_node
            if node_rate <= 0:
                continue
            xfers_per_s = max(fl.count_per_step, 1e-9) / max(workload.step_time_s, 1e-12)
            t = _event_times(xfers_per_s, duration_s, fl.interval_cv, rng)
            if t.size == 0:
                continue
            b = _sizes(t.size, node_rate * duration_s / t.size, fl.size_cv, rng)
            if fl.kind in SYMMETRIC_KINDS:
                dirs = ("tx", "rx")                     # sends what it receives
            elif fl.kind == "p2p":
                dirs = ("tx", "rx")                     # ring pipeline: every stage does both
            elif fl.kind == "kv_push":
                dirs = ("tx",) if role == "prefill" else ("rx",) if role == "decode" else ()
            else:
                dirs = ("tx", "rx")
            for d in dirs:
                per_fabric[fabric][d][0].append(t)
                per_fabric[fabric][d][1].append(b)

        # ── sample the resulting cumulative counters ─────────────────────
        period = 1.0 / coll.sample_hz
        n_s = int(duration_s * coll.sample_hz)
        t_mono = np.arange(n_s) * period
        if coll.sample_jitter_frac > 0:
            t_mono = np.maximum.accumulate(
                t_mono + rng.normal(0, coll.sample_jitter_frac * period, n_s))
        offset_ns = int(rng.normal(0, coll.clock_skew_sd_ns))

        for fabric, spec in (("nvlink", (coll.nvlink_unit, coll.nvlink_width, 450e9 * topo.gpus_per_node)),
                             ("ib",     (coll.ib_unit,     coll.ib_width,     50e9 * topo.gpus_per_node))):
            unit, width, peak = spec
            # each direction of the link carries at most `peak` bytes/s
            d = per_fabric[fabric]
            if not d["tx"][0] and not d["rx"][0]:
                continue
            key = f"{node}/{'nvlink0' if fabric=='nvlink' else 'mlx5_0:1'}/{fabric}"
            links.append(Link(link_key=key, node_id=node,
                              dev_id="nvlink0" if fabric == "nvlink" else "mlx5_0",
                              link_type=fabric,
                              source="dcgm:1011/1012" if fabric == "nvlink" else "sysfs:port_xmit_data",
                              unit=unit, width_bits=width, link_id=-1,
                              peak_bw_bytes_per_s=peak))
            cum = {}
            for d_name in ("tx", "rx"):
                et = np.concatenate(d[d_name][0]) if d[d_name][0] else np.empty(0)
                eb = np.concatenate(d[d_name][1]) if d[d_name][1] else np.empty(0)
                # rx is read read_dur_ns AFTER tx -- a real agent cannot read both at once
                st = t_mono + (coll.read_dur_ns / 1e9 if d_name == "rx" else 0.0)
                cum[d_name] = _cumulative_at(st, et, eb, peak, duration_s)
                truth[f"{key}/{d_name}"] = float(eb.sum())
            mult = UNITS[unit]
            mask = np.uint64((1 << width) - 1)   # 2**64 overflows int64; mask instead
            enc = lambda c: np.floor(c / mult).astype(np.uint64) & mask
            rows.append(pd.DataFrame({
                "t_mono_ns": (t_mono * 1e9).astype(np.int64),
                "t_wall_ns": (t_mono * 1e9).astype(np.int64) + offset_ns,
                "link_key": key,
                "tx_raw": enc(cum["tx"]), "rx_raw": enc(cum["rx"]),
                "read_dur_ns": np.int32(coll.read_dur_ns),
                "flags": np.uint8(0)}))
        if ni == 0 or True:
            pass

    obs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[f.name for f in __import__("commtrace.schema", fromlist=["OBS_SCHEMA"]).OBS_SCHEMA])
    man = Manifest(trace_id=trace_id, sample_hz_nominal=coll.sample_hz,
                   nodes={f"n{i:02d}": {"clock_offset_ns": 0, "offset_method": "ntp",
                                        "offset_uncertainty_ns": int(coll.clock_skew_sd_ns)}
                          for i in range(topo.n_nodes)},
                   ground_truth=({"label": workload.cls, "workload": workload.meta,
                                  "roles": roles} if with_label else None))
    path = write_trace(out_dir, obs, links, man)
    return path, truth
