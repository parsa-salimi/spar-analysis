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


# A flow firing far faster than any sampler we study cannot be resolved by it:
# decode-phase tensor parallelism does 128 all-reduces every 0.72 ms, which over
# a 240 s trace is 43 million transfers. Enumerating them is intractable and
# pointless -- at 50 Hz that traffic is a continuous stream. Above this many
# transfers, they are bundled into evenly-spaced groups carrying the same total
# bytes. Byte totals are exact; only sub-bundle structure is lost, and no
# sampler in the study could have seen it.
MAX_EVENTS = 100_000


def _event_times(rate_per_s: float, duration: float, interval_cv: float,
                 rng: np.random.Generator) -> np.ndarray:
    """When transfers happen. interval_cv >= 0.9 is treated as a Poisson
    arrival process (request-driven); anything lower is a jittered periodic
    loop boundary."""
    if rate_per_s <= 0:
        return np.empty(0)
    if rate_per_s * duration > MAX_EVENTS:      # bundle: see MAX_EVENTS above
        rate_per_s = MAX_EVENTS / duration
        interval_cv = min(interval_cv, 0.01)
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
    # Resolution of the integration grid. Capped at MAX_BINS: without a cap a
    # 300 s trace with sub-millisecond transfers allocates millions of bins per
    # direction per fabric per node and exhausts memory. 0.75 ms at 300 s is
    # still ~90x finer than the fastest sampling we study.
    MAX_BINS = 400_000
    fine = max(min(dur.min() / 4.0, 1e-3), duration / MAX_BINS)
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


@dataclass
class Events:
    """The TRUE transfer events, before any instrument touches them.

    Separating this from sampling is what makes a measurement-artifact study
    possible: the traffic is built once, and each collection setting observes
    the SAME underlying events. Any difference in the result is then
    attributable to the instrument rather than to a different random draw.
    """
    cls: str
    label: str
    step_time_s: float
    duration_s: float
    topo: Topology
    roles: list[str]
    meta: dict
    per_node: list[dict]          # [{fabric: {"tx": (t, b), "rx": (t, b)}}]
    seed: int = 0


def build_events(workload, *, duration_s: float = 300.0,
                 topo: Topology | None = None, seed: int = 0,
                 rank_skew_s: float = 1e-3) -> Events:
    """Lay down every transfer the workload performs, per node."""
    topo = topo or Topology()
    rng = np.random.default_rng(seed)
    roles = topo.resolved(workload.cls)

    # A COLLECTIVE IS A SYNCHRONISED EVENT. Every participating rank enters it
    # together -- that is what makes it a collective. So its event times are
    # drawn ONCE for the job and shared by every node, with only a small
    # per-rank skew on top. Drawing them independently per node (which is what
    # this did first) destroys the synchrony and would make clock skew look
    # harmless in the artifact study, because there would be no synchrony left
    # to lose. Request-driven traffic (kv_push) stays independent per node,
    # because it genuinely is.
    shared_t: dict[int, np.ndarray] = {}
    for fi, fl in enumerate(workload.flows):
        if fl.kind == "kv_push":
            continue
        xps = max(fl.count_per_step, 1e-9) / max(workload.step_time_s, 1e-12)
        shared_t[fi] = _event_times(xps, duration_s, fl.interval_cv,
                                    np.random.default_rng(seed * 1000 + fi))
    # Ranks do not enter a collective perfectly together, but NCCL synchronises
    # tightly -- this is a millisecond-scale effect, not a fraction of the step.
    # Expressing it as a fraction of the inter-transfer interval (as this first
    # did) made it scale with step time, which is wrong and made long-step
    # workloads look far less synchronised than they are.
    RANK_SKEW_S = rank_skew_s

    per_node = []
    for node, role in zip([f"n{i:02d}" for i in range(topo.n_nodes)], roles):
        per_fabric = {"nvlink": {"tx": [[], []], "rx": [[], []]},
                      "ib":     {"tx": [[], []], "rx": [[], []]}}
        for fi, fl in enumerate(workload.flows):
            fabric = "nvlink" if fl.fabric == "intra" else "ib"
            node_rate = fl.bytes_per_s * topo.gpus_per_node
            if node_rate <= 0:
                continue
            xfers_per_s = max(fl.count_per_step, 1e-9) / max(workload.step_time_s, 1e-12)
            if fi in shared_t:
                t = shared_t[fi]
                if t.size:
                    t = np.sort(t + rng.normal(0, RANK_SKEW_S, t.size))
            else:
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
        cat = lambda v: np.concatenate(v) if v else np.empty(0)
        per_node.append({f: {d: (cat(per_fabric[f][d][0]), cat(per_fabric[f][d][1]))
                             for d in ("tx", "rx")} for f in per_fabric})
    return Events(cls=workload.cls, label=workload.label,
                  step_time_s=workload.step_time_s, duration_s=duration_s,
                  topo=topo, roles=roles, meta=dict(workload.meta),
                  per_node=per_node, seed=seed)


def sample_events(ev: Events, out_dir, *, coll: Collection | None = None,
                  trace_id: str | None = None, with_label: bool = True):
    """Observe an Events through one collection setting. Returns (path, truth)."""
    coll = coll or Collection()
    topo, duration_s = ev.topo, ev.duration_s
    rng = np.random.default_rng(coll.seed + 7919)
    trace_id = trace_id or hashlib.sha1(
        f"{ev.label}|{duration_s}|{ev.seed}|{coll}".encode()).hexdigest()[:12]
    links: list[Link] = []
    rows, truth = [], {}

    for ni, node in enumerate([f"n{i:02d}" for i in range(topo.n_nodes)]):
        per_fabric = ev.per_node[ni]
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
            if d["tx"][0].size == 0 and d["rx"][0].size == 0:
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
                et, eb = d[d_name]
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
    obs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[f.name for f in __import__("commtrace.schema", fromlist=["OBS_SCHEMA"]).OBS_SCHEMA])
    man = Manifest(trace_id=trace_id, sample_hz_nominal=coll.sample_hz,
                   nodes={f"n{i:02d}": {"clock_offset_ns": 0, "offset_method": "ntp",
                                        "offset_uncertainty_ns": int(coll.clock_skew_sd_ns)}
                          for i in range(topo.n_nodes)},
                   ground_truth=({"label": ev.cls, "workload": ev.meta,
                                  "roles": ev.roles} if with_label else None))
    path = write_trace(out_dir, obs, links, man)
    return path, truth


def synth_trace(workload, out_dir, *, duration_s: float = 300.0,
                topo: Topology | None = None, coll: Collection | None = None,
                trace_id: str | None = None, with_label: bool = True):
    """Convenience: build the events and observe them in one step."""
    coll = coll or Collection()
    ev = build_events(workload, duration_s=duration_s, topo=topo, seed=coll.seed)
    return sample_events(ev, out_dir, coll=coll, trace_id=trace_id,
                         with_label=with_label)
