"""Rung A1 of the symmetry attack: constant-rate partition padding.

Threat model, capabilities and cost are fixed in comm_model/ATTACK_SYMMETRY.md.
This module only transforms the TRUE traffic (an Events object) before any
instrument observes it, so clean and padded versions of a trace differ by the
attack and nothing else.

The adversary splits the nodes into senders P and receivers D and streams dummy
bytes P -> D over the inter-node fabric at a constant rate. To reach node-level
symmetry s* on a node whose real inter-node traffic is B each way, it needs

    rho = B (1/s* - 1)

Bytes are conserved: everything P sends, D receives, split evenly. Only the
inter-node fabric is padded -- intra-node NVLink traffic cannot be observed from
outside the node in a way that would make padding it useful, and it is not what
the symmetry feature reads.

The attack pays its cost inside the trace. The generator does not model link
contention, so naively adding padding on top of a busy link produces rates
above line rate -- physically impossible, and trivially detectable for the wrong
reason. Instead: if a node's real traffic plus padding would exceed
MAX_UTIL x line rate in one direction, the whole job is slowed by the factor
needed to make it fit. Every real event is stretched in time by that factor
(fewer steps per second, same bytes per step) and the padding is scaled down with
it, so the target symmetry is preserved and the link runs at MAX_UTIL. This is
the wire-bound case of the analytic cost model in comm_model/attack_symmetry.py,
and the slowdown applied is recorded on the returned Events as meta["pad_slowdown"].
"""
from __future__ import annotations
import copy
import numpy as np

from . import _paths  # noqa: F401
from commtrace.gen import Events


IB_CAP_NODE = 50e9 * 8      # bytes/s per direction per node: 8 GPUs x 50 GB/s NIC
MAX_UTIL = 0.95


def _stretch(per_node, f: float, dur: float):
    """Slow the whole job by factor f: event times scale, bytes per event do not."""
    for node in per_node:
        for fab in node:
            for d in ("tx", "rx"):
                t, b = node[fab][d]
                t2 = t * f
                keep = t2 < dur
                node[fab][d] = (t2[keep], b[keep])


def pad_partition(ev: Events, s_target: float = 0.10, spacing_s: float = 0.0005,
                  seed: int = 0) -> Events:
    n = len(ev.per_node)
    P, D = list(range(n // 2)), list(range(n // 2, n))
    k = 1.0 / s_target - 1.0
    dur = ev.duration_s

    def real_rate(i):
        ib = ev.per_node[i]["ib"]
        return 0.5 * (ib["tx"][1].sum() + ib["rx"][1].sum()) / dur

    B = np.array([real_rate(i) for i in range(n)])
    rho = np.zeros(n)
    rho[P] = k * B[P]
    sent = rho[P].sum()
    rho[D] = sent / len(D)
    # demand in the padded direction, per node, before any slowdown
    demand = B + rho
    f = max(1.0, float(demand.max() / (MAX_UTIL * IB_CAP_NODE)))

    per_node = copy.deepcopy(ev.per_node)
    if f > 1.0:
        _stretch(per_node, f, dur)
    rho = rho / f

    # A real constant stream at tens of GB/s is millions of packets per second:
    # perfectly smooth at any rate we sample. Spacing is therefore set well below
    # the generator's integration grid and left unjittered. Coarser spacing makes
    # the number of pad transfers caught per sample interval fluctuate, and with
    # padding ~9x the real traffic that fluctuation masquerades as bursts.
    def stream(total_Bps):
        t = np.arange(0.0, dur, spacing_s)
        return t, np.full(t.size, total_Bps * spacing_s)

    for i in range(n):
        if rho[i] <= 0:
            continue
        d = "tx" if i in P else "rx"
        t, b = stream(rho[i])
        et, eb = per_node[i]["ib"][d]
        per_node[i]["ib"][d] = (np.concatenate([et, t]), np.concatenate([eb, b]))

    out = copy.copy(ev)
    out.per_node = per_node
    out.step_time_s = ev.step_time_s * f
    out.meta = {**ev.meta, "pad_slowdown": f, "pad_s_target": s_target}
    out.label = f"{ev.label}+padA1(s={s_target})"
    return out
