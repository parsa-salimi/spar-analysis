"""Read a comm_trace and turn raw counters into the features a detector may use.

All the interpretation the agent deliberately did not do happens here:
un-wrapping, unit conversion, rate computation, and the check that a counter did
not wrap more than once between samples (which cannot be detected from the
counter alone -- only by comparing the implied rate to the link's line rate).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .schema import UNITS, FLAG_WRAP_SUSPECT, read_trace


def preflight(trace_dir) -> pd.DataFrame:
    """Decide, from links.json and the nominal sample rate alone, whether each
    link CAN be read safely -- before looking at a single value. This is the
    check that actually protects us; see the note in rates()."""
    _, links, man = read_trace(trace_dir)
    dt = 1.0 / man.sample_hz_nominal
    return pd.DataFrame([dict(
        link_key=l.link_key, link_type=l.link_type, unit=l.unit,
        width_bits=l.width_bits, wrap_seconds=l.wrap_seconds,
        sample_interval_s=dt, min_safe_hz=l.min_safe_hz(),
        safe=dt < l.wrap_seconds / 4) for l in links])


def rates(trace_dir) -> pd.DataFrame:
    """One row per (link, sample interval): tx and rx byte rates."""
    obs, links, man = read_trace(trace_dir)
    meta = {l.link_key: l for l in links}
    out = []
    for key, g in obs.groupby("link_key", sort=False):
        L = meta[key]
        g = g.sort_values("t_mono_ns")
        mask = np.uint64((1 << L.width_bits) - 1)
        dt = np.diff(g.t_mono_ns.values) / 1e9
        mult = UNITS[L.unit]
        rec = {"link_key": key, "node_id": L.node_id, "link_type": L.link_type,
               "t_mono_s": g.t_mono_ns.values[1:] / 1e9,
               "t_wall_s": g.t_wall_ns.values[1:] / 1e9, "dt": dt}
        for d in ("tx", "rx"):
            v = g[f"{d}_raw"].values.astype(np.uint64)
            # (v2 - v1) mod 2**width. Correct for at most ONE wrap per interval;
            # beyond that it is wrong rather than missing, hence the check below.
            delta = ((v[1:] - v[:-1]) & mask).astype(np.float64) * mult
            rec[f"{d}_bytes"] = delta
            rec[f"{d}_Bps"] = np.divide(delta, dt, out=np.zeros_like(delta),
                                        where=dt > 0)
        df = pd.DataFrame(rec)
        # Two independent checks, and the second is the one that matters.
        #
        # (a) DATA check: a rate above line rate can only mean a wrap we missed.
        #     Necessary, but it only catches a SINGLE wrap. Once the counter can
        #     wrap more than once per interval the modular delta is an arbitrary
        #     value in [0, 2**width * unit), which divided by dt is BELOW line
        #     rate -- so this check is blind exactly when the failure is certain.
        #     Measured on synthetic traces: a 32-bit word counter on NVLink
        #     sampled at 10 Hz recovers 13% of the true bytes and trips no alarm.
        #
        # (b) CONFIG check: the counter's whole range corresponds to
        #     `wrap_seconds` at line rate. If the sample interval exceeds that,
        #     the reading is untrustworthy whatever value it holds. This is
        #     decidable before collection starts, from links.json alone.
        df["wrap_rate_exceeded"] = ((df.tx_Bps > L.peak_bw_bytes_per_s * 1.05) |
                                    (df.rx_Bps > L.peak_bw_bytes_per_s * 1.05))
        df["wrap_undersampled"] = df.dt > L.wrap_seconds
        df["wrap_suspect"] = df.wrap_rate_exceeded | df.wrap_undersampled
        df["peak_bw"] = L.peak_bw_bytes_per_s
        df["wrap_seconds"] = L.wrap_seconds
        out.append(df)
    r = pd.concat(out, ignore_index=True)
    r.attrs["manifest"] = man
    return r


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) < lag + 3:
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    return 0.0 if sa < 1e-12 or sb < 1e-12 else float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _fft_peak(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 8 or x.std() < 1e-12:
        return 0.0, 0.0
    y = x - x.mean()
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    p[0] = 0.0
    k = int(np.argmax(p))
    total = p.sum()
    return (len(x) / k if k else 0.0), float(p[k] / total) if total > 0 else 0.0


def window_features(r: pd.DataFrame, window_s: float = 30.0,
                    stride_s: float = 15.0) -> pd.DataFrame:
    """Per (node, fabric, window) features. Nothing here uses anything a
    cumulative byte counter cannot provide."""
    rows = []
    for (node, ltype), g in r.groupby(["node_id", "link_type"], sort=False):
        g = g.sort_values("t_mono_s")
        t0, t1 = g.t_mono_s.min(), g.t_mono_s.max()
        w = t0
        while w + window_s <= t1 + 1e-9:
            c = g[(g.t_mono_s >= w) & (g.t_mono_s < w + window_s)]
            if len(c) >= 4:
                tx, rx = c.tx_Bps.values, c.rx_Bps.values
                tot = tx + rx
                mx, mn = np.maximum(tx, rx), np.minimum(tx, rx)
                sym = float(np.divide(mn.sum(), mx.sum()) if mx.sum() > 0 else 1.0)
                per, pw = _fft_peak(tot)
                rows.append(dict(
                    node_id=node, link_type=ltype, w_start=w,
                    tx_Bps=float(tx.mean()), rx_Bps=float(rx.mean()),
                    total_Bps=float(tot.mean()),
                    symmetry=sym,
                    cv=float(tot.std() / tot.mean()) if tot.mean() > 0 else 0.0,
                    acf1=_acf(tot, 1), acf2=_acf(tot, 2), acf5=_acf(tot, 5),
                    fft_period=per, fft_peak_frac=pw,
                    utilisation=float(tot.mean() / c.peak_bw.iloc[0]),
                    wrap_suspect_frac=float(c.wrap_suspect.mean()),
                    n_samples=len(c)))
            w += stride_s
    return pd.DataFrame(rows)


def node_window_features(r: pd.DataFrame, **kw) -> pd.DataFrame:
    """One row per (node, window), fabrics side by side -- what an aggregator
    would hand the classifier."""
    wf = window_features(r, **kw)
    if wf.empty:
        return wf
    idx = ["node_id", "w_start"]
    wide = wf.pivot_table(index=idx, columns="link_type",
                          values=["total_Bps", "symmetry", "cv", "acf1", "fft_peak_frac",
                                  "utilisation", "wrap_suspect_frac"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()
