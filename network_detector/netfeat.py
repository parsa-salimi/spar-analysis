"""Network features: what a provider sees from each node's NIC counters alone.

Input: the NIC sidecar logs (`scripts/collect_host_telemetry.py` on the branch),
one row per second per node, tx and rx kept separately, in MiB/s.

Output: one row per (job, window), 30 s windows at a 15 s stride on a grid shared
by both nodes -- the same windowing as the per-GPU detector, so the two can be
joined and compared window for window.

Per node, per window
  vol        log10(1 + mean(tx + rx))          how much moves
  sym        min(sum tx, sum rx) / max(...)     is it balanced
  dir        (sum tx - sum rx) / (sum tx + sum rx), signed
  duty       fraction of seconds above 1 MiB/s  is the link busy all the time
  cv, acf1, acf3, fft_peak_frac                 how uneven / smooth / periodic
  burst_sym  symmetry of the part above the window's 10th percentile
  peak_mean  p95 / mean                         how spiky

Across the two nodes (when both logs exist)
  xcorr      correlation of node0 tx with node1 tx   do they move in lockstep
  xdir       correlation of node0 tx with node1 rx   does 0 send what 1 receives
  anti_dir   -dir0 * dir1: large when one node only sends and the other only
             receives, as in a two-stage pipeline
  pair_sym   (tx0 + tx1) vs (rx0 + rx1): conservation across the job

Absolute byte volume on this cluster reflects one TCP gVNIC per node, not a
production fabric, so every analysis reports results with and without `vol`.
"""
from __future__ import annotations
import numpy as np, pandas as pd

WIN, STRIDE = 30.0, 15.0
ACTIVE_MIBPS = 1.0


def _acf(x, lag):
    if len(x) <= lag + 2:
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    return 0.0 if sa < 1e-9 or sb < 1e-9 else float(np.corrcoef(a, b)[0, 1])


def _fft_peak_frac(x):
    if len(x) < 8 or x.std() < 1e-9:
        return 0.0
    p = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
    p[0] = 0.0
    return float(p.max() / p.sum()) if p.sum() > 0 else 0.0


def _sym(a, b):
    hi = max(a, b)
    return float(min(a, b) / hi) if hi > 0 else 1.0


def node_window(tx, rx) -> dict:
    tot = tx + rx
    st, sr = tx.sum(), rx.sum()
    bt, br = np.percentile(tx, 10), np.percentile(rx, 10)
    ut, ur = np.clip(tx - bt, 0, None).sum(), np.clip(rx - br, 0, None).sum()
    m = tot.mean()
    return dict(
        vol=float(np.log10(1 + m)),
        sym=_sym(st, sr),
        dir=float((st - sr) / (st + sr)) if (st + sr) > 0 else 0.0,
        duty=float((tot > ACTIVE_MIBPS).mean()),
        cv=float(tot.std() / m) if m > 0 else 0.0,
        acf1=_acf(tot, 1), acf3=_acf(tot, 3),
        fft_peak_frac=_fft_peak_frac(tot),
        burst_sym=_sym(ut, ur) if (ut + ur) > 0.02 * (st + sr) else 1.0,
        peak_mean=float(np.percentile(tot, 95) / m) if m > 0 else 0.0,
    )


NODE_COLS = ["vol", "sym", "dir", "duty", "cv", "acf1", "acf3",
             "fft_peak_frac", "burst_sym", "peak_mean"]
CROSS_COLS = ["xcorr", "xdir", "anti_dir", "pair_sym"]


def _safe_corr(a, b):
    if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def job_windows(nodes: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """nodes: {node_rank: NIC frame}. Returns one row per window."""
    for n, d in nodes.items():
        nodes[n] = d.sort_values("timestamp_epoch").reset_index(drop=True)
    t0 = min(d.timestamp_epoch.iloc[0] for d in nodes.values())
    t1 = max(d.timestamp_epoch.iloc[-1] for d in nodes.values())
    # second-resolution grid shared by both nodes, for the cross-node features
    grid = {n: d.assign(s=np.floor(d.timestamp_epoch - t0).astype(int))
                 .groupby("s")[["nic_tx_mbps", "nic_rx_mbps"]].mean()
            for n, d in nodes.items()}
    rows, w = [], t0
    while w + WIN <= t1 + 1:
        r = dict(w_start=w - t0, w_abs=w)
        per = {}
        for n, d in nodes.items():
            c = d[(d.timestamp_epoch >= w) & (d.timestamp_epoch < w + WIN)]
            if len(c) >= 10:
                per[n] = node_window(c.nic_tx_mbps.values, c.nic_rx_mbps.values)
        if 0 in per:
            vals = {k: np.array([per[n][k] for n in per]) for k in NODE_COLS}
            for k in NODE_COLS:
                if k == "dir":
                    r["dir_abs"] = float(np.abs(vals[k]).mean())
                else:
                    r[k] = float(vals[k].mean())
            r["n_nodes"] = len(per)
            if len(per) >= 2:
                s0, s1 = int(w - t0), int(w - t0 + WIN)
                g0 = grid[0].loc[s0:s1 - 1]; g1 = grid[1].loc[s0:s1 - 1]
                j = g0.join(g1, lsuffix="0", rsuffix="1", how="inner")
                r["xcorr"] = _safe_corr(j.nic_tx_mbps0.values, j.nic_tx_mbps1.values)
                r["xdir"] = _safe_corr(j.nic_tx_mbps0.values, j.nic_rx_mbps1.values)
                r["anti_dir"] = float(-per[0]["dir"] * per[1]["dir"])
                tx = j.nic_tx_mbps0.sum() + j.nic_tx_mbps1.sum()
                rx = j.nic_rx_mbps0.sum() + j.nic_rx_mbps1.sum()
                r["pair_sym"] = _sym(tx, rx)
            rows.append(r)
        w += STRIDE
    return pd.DataFrame(rows)
