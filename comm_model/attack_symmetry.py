"""Cost surface for the partition-padding attack on the tx/rx symmetry feature.

Threat model, adversary capabilities and cost metric are fixed in advance in
ATTACK_SYMMETRY.md; this script only measures. Read that first.

The adversary splits nodes into a sending set and a receiving set and injects
dummy traffic between them, so that node-level symmetry min(tx,rx)/max(tx,rx)
falls to a target s*. Reaching s* requires padding at k = 1/s* - 1 times the
job's real inter-node rate. The cost is the wall-clock slowdown that padding
imposes once the padded traffic no longer fits in the step.

The cost model is deliberately generous to the adversary: padding may be
scheduled into the idle gaps between collectives, so it costs nothing until the
TOTAL per-step inter-node bytes exceed what the link carries in one step.
"""
from __future__ import annotations
import numpy as np, pandas as pd

import commvol as C

TARGETS = (0.50, 0.20, 0.10, 0.05)


def load() -> pd.DataFrame:
    g = pd.read_csv("grid.csv.gz", low_memory=False)
    t = g[(g.cls == "training") & g.feasible & (~g.single_node) & (g.inter_bps > 0)].copy()
    gpn = t.hw.map({k: v.gpus_per_node for k, v in C.HARDWARE.items()})
    nic = t.hw.map({k: v.nic_bw for k, v in C.HARDWARE.items()})
    # Per NODE, one direction. commvol reports per-GPU rates.
    t["cap_node"]   = nic * gpn
    t["inter_node"] = t.inter_bps * gpn
    t["bytes_step"] = t.inter_node * t.step_time_s
    # What the step would take if the NIC were infinitely fast: the floor that
    # padding has to exceed before it costs anything.
    t["floor_s"] = np.maximum(
        t.compute_time_s,
        t.intra_bps * gpn * t.step_time_s / (450e9 * gpn))
    return t


def slowdown(t: pd.DataFrame, s_star: float) -> pd.Series:
    """Wall-clock multiple imposed by padding to symmetry s_star."""
    k = 1.0 / s_star - 1.0
    wire = (1.0 + k) * t.bytes_step / t.cap_node
    return np.maximum(t.floor_s, wire) / t.step_time_s


def cost_surface(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in TARGETS:
        sl = slowdown(t, s)
        rows.append(dict(target_symmetry=s, padding_multiple=1 / s - 1,
                         frac_free=float((sl <= 1.0001).mean()),
                         median_slowdown=float(sl.median()),
                         p90_slowdown=float(sl.quantile(0.90))))
    return pd.DataFrame(rows)


def by_regime(t: pd.DataFrame, s_star: float = 0.10) -> pd.DataFrame:
    sl = slowdown(t, s_star)
    masks = {
        "compute-bound": ~t.comm_bound.astype(bool),
        "comm-bound":     t.comm_bound.astype(bool),
        "DDP":            t.sharding == "ddp",
        "FSDP":           t.sharding == "fsdp",
        "world<=32":      t.world <= 32,
        "world>=128":     t.world >= 128,
    }
    return pd.DataFrame([
        dict(regime=name, n=int(m.sum()),
             frac_free=float((sl[m] <= 1.0001).mean()),
             median_slowdown=float(sl[m].median()))
        for name, m in masks.items()])


if __name__ == "__main__":
    t = load()
    print(f"feasible multi-node training configurations: {len(t)}\n")
    cs = cost_surface(t)
    print(cs.round(3).to_string(index=False))
    print()
    print(by_regime(t).round(3).to_string(index=False))
    cs.to_csv("results_attack_cost.csv", index=False)
    by_regime(t).to_csv("results_attack_regime.csv", index=False)
    print("\nwrote results_attack_cost.csv, results_attack_regime.csv")
