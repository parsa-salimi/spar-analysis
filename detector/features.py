"""Detector features: the reader's window features plus a baseline/burst split.

Why split a rate series into baseline and burst
-----------------------------------------------
A node's transmit rate over a window is the sum of two things with different
origins. Collectives -- all-reduce, all-gather, all-to-all -- arrive as BURSTS:
the job computes, then every rank communicates at once, then computes again.
Anything steady -- a background stream, a saturated link, or an adversary's
padding -- sits underneath as a BASELINE.

The partition-padding attack (comm_model/ATTACK_SYMMETRY.md, rung A1) adds a
constant one-way stream. That raises the baseline of one direction and leaves
the bursts untouched. So symmetry computed on the bursts alone is unmoved by the
attack, while symmetry on the raw totals is exactly what the attack targets.

  baseline  = a low percentile of the rate within the window, per direction
  burst     = max(rate - baseline, 0)

Features added here, per (node, fabric, window), then averaged over nodes:

  burst_sym     min(burst tx, burst rx) / max(...)   -- the D1 defence
  base_asym     |base tx - base rx| / (base tx + base rx)
                -- where a constant one-way stream shows up. Attacker-controlled
                   by construction, so it belongs in NO attack-robust feature set;
                   it is computed because it is the attack's fingerprint.
  burst_Bps     mean burst rate, tx + rx -- the collective traffic with any
                steady floor removed

Everything is computable from a cumulative byte counter. Nothing uses message
sizes, peer identities or anything reported from inside the workload.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from . import _paths  # noqa: F401
from commtrace.reader import rates, node_window_features, cross_node_features, FEATURE_COLS

BASELINE_PCT = 10.0

BURST_COLS = [f"{f}_{fab}" for f in ("burst_sym", "base_asym", "burst_Bps")
              for fab in ("ib", "nvlink")]


# A window whose burst component is below this fraction of its total traffic has
# no bursts worth the name -- only sampling noise riding on a steady floor -- and
# the ratio of two noise terms carries no directional information.
MIN_BURST_FRAC = 0.02


def _burst_window(tx: np.ndarray, rx: np.ndarray) -> dict:
    bt, br = np.percentile(tx, BASELINE_PCT), np.percentile(rx, BASELINE_PCT)
    ut, ur = np.clip(tx - bt, 0, None), np.clip(rx - br, 0, None)
    st, sr = ut.sum(), ur.sum()
    hi = max(st, sr)
    total = tx.sum() + rx.sum()
    # No bursts: uninformative. 1.0 matches the reader's convention for the raw
    # symmetry of an idle window.
    if hi <= 0 or (total > 0 and (st + sr) < MIN_BURST_FRAC * total):
        burst_sym = 1.0
    else:
        burst_sym = float(min(st, sr) / hi)
    base_asym = float(abs(bt - br) / (bt + br)) if (bt + br) > 0 else 0.0
    return dict(burst_sym=burst_sym, base_asym=base_asym,
                burst_Bps=float((ut + ur).mean()))


def burst_features(r: pd.DataFrame, window_s: float = 30.0, stride_s: float = 15.0,
                   t_origin: float | None = None) -> pd.DataFrame:
    """One row per (node, window), fabrics side by side. Uses the same integer
    window grid as reader.window_features so the tables join exactly."""
    origin = r.t_mono_s.min() if t_origin is None else t_origin
    rows = []
    for (node, fab), g in r.groupby(["node_id", "link_type"], sort=False):
        g = g.sort_values("t_mono_s")
        t1 = g.t_mono_s.max()
        w = origin + np.floor((g.t_mono_s.min() - origin) / stride_s) * stride_s
        while w + window_s <= t1 + 1e-9:
            c = g[(g.t_mono_s >= w) & (g.t_mono_s < w + window_s)]
            if len(c) >= 4:
                d = _burst_window(c.tx_Bps.values, c.rx_Bps.values)
                d.update(node_id=node, link_type=fab,
                         w_idx=int(round((w - origin) / stride_s)))
                rows.append(d)
            w += stride_s
    if not rows:
        return pd.DataFrame(columns=["node_id", "w_idx"] + BURST_COLS)
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["node_id", "w_idx"], columns="link_type",
                          values=["burst_sym", "base_asym", "burst_Bps"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()


def detector_features(trace_dir, window_s: float = 30.0, stride_s: float = 15.0
                      ) -> pd.DataFrame:
    """One row per window: node-averaged reader features, cross-node synchrony,
    and the burst decomposition."""
    r = rates(trace_dir)
    origin = r.t_mono_s.min()
    nw = node_window_features(r, window_s=window_s, stride_s=stride_s, t_origin=origin)
    if nw.empty:
        return nw
    bw = burst_features(r, window_s=window_s, stride_s=stride_s, t_origin=origin)
    nw = nw.merge(bw, on=["node_id", "w_idx"], how="left")
    agg = nw.groupby("w_idx").mean(numeric_only=True).reset_index()
    xn = cross_node_features(r, window_s=window_s, stride_s=stride_s,
                             t_origin=r.t_wall_s.min())
    out = agg.merge(xn[["w_idx", "xnode_corr"]], on="w_idx", how="left")
    out["xnode_corr"] = out.xnode_corr.fillna(0.0)
    for c in FEATURE_COLS + BURST_COLS:
        if c not in out.columns:
            # absent fabric: no traffic, so balanced and empty
            out[c] = 1.0 if c.startswith(("symmetry", "burst_sym")) else 0.0
    return out
