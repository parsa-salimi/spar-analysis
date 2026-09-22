"""Does the synthetic generator stand in for a real trace?

With compute access permanently unavailable, every result on this side of the
project is derived on synthetic traces. That makes the generator's fidelity a
load-bearing assumption, and an assumption that has never been checked. This is
the check, written before any real data exists so the tolerances cannot be
chosen to fit what arrives.

The comparison is between a REAL trace and a synthetic trace generated from the
same workload configuration and observed through the same collection settings.
Each check reports the measured values, whether it falls inside a tolerance fixed
in advance, and -- if it does not -- what that implies and what to do.

Checks are graded, not pooled into one verdict:

  A  the closed-form model is wrong if these fail, and everything downstream
     inherits the error
  B  the generator is a simplification; detection numbers must be re-derived on
     real traces, but the architecture stands
  C  instrument properties; these calibrate the collection settings rather than
     the model

Usage:
    python3 validate.py <real_trace_dir> <synthetic_trace_dir>
    python3 validate.py --selftest
"""
from __future__ import annotations
import sys, json
from dataclasses import dataclass
from pathlib import Path

import numpy as np, pandas as pd

import config  # noqa: F401  (puts the repo on sys.path)
from commtrace.reader import rates, window_features, cross_node_features, preflight
from commtrace.schema import read_trace


# ── tolerances, FIXED IN ADVANCE ───────────────────────────────────────────
# Rationale for each is in VALIDATION.md. They are deliberately asymmetric:
# structural properties (symmetry, fabric assignment) get tight bounds because
# they follow from what a collective IS; volume gets a loose one because real
# NCCL carries protocol overhead, switches to tree algorithms for small
# messages, and chunks transfers.
@dataclass(frozen=True)
class Check:
    key: str
    tier: str
    what: str
    lo: float
    hi: float
    on_fail: str

CHECKS = [
    Check("bytes_total_ratio", "A", "total bytes moved, real / analytic",
          0.80, 1.35,
          "commvol's per-step byte formulas are wrong; every volume figure and "
          "the evasion cost model inherit the error"),
    Check("inter_share", "A", "share of bytes crossing the NIC, real vs synthetic",
          0.70, 1.43,
          "the fabric assignment is wrong -- NCCL is not routing collectives the "
          "way the model assumes; the intra/inter split and the whole "
          "inter-node analysis need redoing"),
    Check("symmetry_inter_train", "A", "tx/rx symmetry of training inter-node traffic",
          0.85, 1.00,
          "the central separability result collapses: symmetry is the feature "
          "that takes held-out-model AUC from 0.76 to 0.97"),
    Check("symmetry_inter_serve", "A", "tx/rx symmetry of disaggregated serving",
          0.00, 0.25,
          "KV-cache push is not directional at node level as modelled; symmetry "
          "does not separate the classes after all"),
    Check("burst_duty_cycle_ratio", "B", "fraction of intervals carrying traffic, real / synth",
          0.50, 2.00,
          "transfers are not ramps of duration size/bandwidth -- likely NCCL "
          "chunking; re-fit _cumulative_at and re-run the artifact study"),
    Check("interval_cv_ratio", "B", "CV of gaps between bursts, real / synth",
          0.40, 2.50,
          "step cadence is less regular than modelled; cadence features will be "
          "weaker on real traces than in the sweep"),
    Check("xnode_corr_train", "B", "cross-node transmit correlation, training",
          0.60, 1.00,
          "collectives are not as synchronised as modelled; the cross-node "
          "feature is weaker and the clock-skew conclusion needs revisiting"),
    Check("rate_p95_over_mean_ratio", "B", "burstiness shape, real / synth",
          0.40, 2.50,
          "within-window rate distribution differs; window statistics "
          "(cv, acf, fft) will behave differently"),
    Check("effective_hz_ratio", "C", "achieved sample rate / nominal",
          0.90, 1.10,
          "the agent is not keeping up with its configured rate"),
    Check("wrap_suspect_frac", "C", "fraction of intervals flagged as wrapping",
          0.00, 0.01,
          "counters are wrapping between reads -- raise the sample rate or move "
          "to 64-bit counters before collecting anything at scale"),
]


# ── measurements ────────────────────────────────────────────────────────────
def _profile(trace_dir) -> dict:
    """Everything the checks need, measured from one trace."""
    r = rates(trace_dir)
    _, links, man = read_trace(trace_dir)
    out = {}
    for fab in ("ib", "nvlink"):
        g = r[r.link_type == fab]
        if g.empty:
            out[f"{fab}_bytes"] = 0.0
            continue
        out[f"{fab}_bytes"] = float(g.tx_bytes.sum() + g.rx_bytes.sum())
        tot = g.tx_Bps.values + g.rx_Bps.values
        live = tot > 0.01 * max(tot.max(), 1e-9)
        out[f"{fab}_duty"] = float(live.mean())
        out[f"{fab}_p95_over_mean"] = float(
            np.percentile(tot, 95) / tot.mean()) if tot.mean() > 0 else 0.0
        gaps = np.diff(np.flatnonzero(live).astype(float))
        out[f"{fab}_interval_cv"] = float(gaps.std() / gaps.mean()) if len(gaps) > 2 and gaps.mean() > 0 else 0.0
        tx, rx = g.tx_Bps.sum(), g.rx_Bps.sum()
        out[f"{fab}_symmetry"] = float(min(tx, rx) / max(tx, rx)) if max(tx, rx) > 0 else 1.0
    out["total_bytes"] = out.get("ib_bytes", 0.0) + out.get("nvlink_bytes", 0.0)
    out["inter_share"] = (out.get("ib_bytes", 0.0) / out["total_bytes"]) if out["total_bytes"] > 0 else 0.0
    xn = cross_node_features(r)
    out["xnode_corr"] = float(xn.xnode_corr.mean()) if len(xn) else 0.0
    out["wrap_suspect_frac"] = float(r.wrap_suspect.mean())
    dur = (r.t_mono_s.max() - r.t_mono_s.min()) if len(r) else 1.0
    n_per_link = len(r) / max(r.link_key.nunique(), 1)
    out["effective_hz"] = float(n_per_link / dur) if dur > 0 else 0.0
    out["nominal_hz"] = float(man.sample_hz_nominal)
    out["cls"] = (man.ground_truth or {}).get("label", "unknown")
    return out


def _ratio(a, b):
    return float(a / b) if b not in (0, 0.0) else (np.nan if a else 1.0)


def compare(real_dir, synth_dir, analytic_bytes: float | None = None) -> pd.DataFrame:
    R, S = _profile(real_dir), _profile(synth_dir)
    is_train = R["cls"] == "training"
    m = {
        "bytes_total_ratio": _ratio(R["total_bytes"], analytic_bytes if analytic_bytes
                                    else S["total_bytes"]),
        "inter_share": _ratio(R["inter_share"], S["inter_share"]),
        "symmetry_inter_train": R.get("ib_symmetry", np.nan) if is_train else np.nan,
        "symmetry_inter_serve": np.nan if is_train else R.get("ib_symmetry", np.nan),
        "burst_duty_cycle_ratio": _ratio(R.get("ib_duty", 0), S.get("ib_duty", 0)),
        "interval_cv_ratio": _ratio(R.get("ib_interval_cv", 0), S.get("ib_interval_cv", 0)),
        "xnode_corr_train": R["xnode_corr"] if is_train else np.nan,
        "rate_p95_over_mean_ratio": _ratio(R.get("ib_p95_over_mean", 0),
                                           S.get("ib_p95_over_mean", 0)),
        "effective_hz_ratio": _ratio(R["effective_hz"], R["nominal_hz"]),
        "wrap_suspect_frac": R["wrap_suspect_frac"],
    }
    rows = []
    for c in CHECKS:
        v = m[c.key]
        applicable = not (isinstance(v, float) and np.isnan(v))
        rows.append(dict(tier=c.tier, check=c.key, what=c.what,
                         measured=v, lo=c.lo, hi=c.hi,
                         verdict=("n/a" if not applicable else
                                  "PASS" if c.lo <= v <= c.hi else "FAIL"),
                         on_fail=c.on_fail))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    out = []
    for tier, g in df.groupby("tier"):
        out.append(f"\n── tier {tier} " + "─"*60)
        for r in g.itertuples():
            v = "  n/a" if r.verdict == "n/a" else f"{r.measured:7.3f}"
            out.append(f"  {r.verdict:5s} {r.check:26s} {v}  "
                       f"[{r.lo:.2f}, {r.hi:.2f}]  {r.what}")
            if r.verdict == "FAIL":
                out.append(f"        -> {r.on_fail}")
    f = df[df.verdict == "FAIL"]
    out.append(f"\n{(df.verdict=='PASS').sum()} pass, {len(f)} fail, "
               f"{(df.verdict=='n/a').sum()} not applicable")
    if len(f):
        worst = sorted(f.tier.unique())[0]
        out.append(f"Worst failing tier: {worst}. See VALIDATION.md for what follows.")
    return "\n".join(out)
