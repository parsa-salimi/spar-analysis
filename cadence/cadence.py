#!/usr/bin/env python3
"""Can step cadence tell one-directional TRAINING from one-directional SERVING?

The question. The lopsided parameter server (attack_workloads/train_ps_lopsided.py)
defeats the symmetry rule: its nodes send one way, like pipeline-serving stages.
What it cannot easily change is its rhythm. A training loop repeats on a fixed
step period -- compute, then communicate, then compute -- while online serving
is driven by requests that arrive at random. This simulates both, observes each
through a byte counter at 1-50 Hz, and measures how well rhythm alone separates
them at each sample rate.

Everything here is a MODEL. The traffic shapes are drawn from stated parameter
ranges; nothing is measured. The output is a feasibility claim and a sample-rate
requirement, to be checked against real traces collected at 10 Hz.

Features are scale-free on purpose -- no absolute byte volume. Volume separates
these classes easily today, but an adversary can shrink it (DiLoCo, compression),
so a defence that rests on it is not a defence against this attacker.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

DT = 1e-3            # simulation grid, s
DUR = 240.0          # trace length, s
WIN, STRIDE = 30.0, 15.0
RATES = [1, 2, 5, 10, 20, 50]
N_PER_FAMILY = 40
OUT = Path(__file__).resolve().parent / "results"


# ── traffic construction ────────────────────────────────────────────────────
class Trace:
    """Bytes crossing one node's NIC per DT bin (both directions summed)."""

    def __init__(self):
        self.b = np.zeros(int(DUR / DT) + 2)

    def add(self, t0: np.ndarray, size: np.ndarray, link: float):
        """Transfers of `size` bytes starting at `t0`, each at `link` bytes/s."""
        t0, size = np.atleast_1d(t0), np.atleast_1d(size)
        dur = size / link
        short = dur <= DT
        i = np.clip((t0[short] / DT).astype(int), 0, len(self.b) - 1)
        np.add.at(self.b, i, size[short])
        for s0, z, d in zip(t0[~short], size[~short], dur[~short]):
            e0 = min(s0 + d, DUR)
            if s0 >= DUR:
                continue
            rate = z / d
            ia, ie = int(s0 / DT), int(e0 / DT)
            if ia == ie:
                self.b[ia] += rate * (e0 - s0)
                continue
            self.b[ia] += rate * ((ia + 1) * DT - s0)
            self.b[ia + 1:ie] += rate * DT
            self.b[ie] += rate * (e0 - ie * DT)


def poisson_times(rate: float, rng) -> np.ndarray:
    n = rng.poisson(rate * DUR * 1.2) + 10
    t = np.cumsum(rng.exponential(1 / rate, n))
    return t[t < DUR]


def fam_ps_step(rng):
    """Lopsided parameter server, sync every step: upload small, download dense."""
    link = rng.choice([12.5e9, 50e9])
    tx = rng.uniform(0.05, 1.0)                  # dense download time per step, s
    tc = rng.uniform(0.2, 2.0)                   # compute time per step, s
    period, jit = tc + tx, rng.uniform(0.01, 0.05)
    n = int(DUR / period) + 2
    starts = np.cumsum(period * (1 + jit * rng.standard_normal(n))) - period * rng.uniform()
    tr = Trace()
    ok = (starts > 0) & (starts < DUR)
    tr.add(starts[ok], np.full(ok.sum(), 0.15 * tx * link), link)       # upload, c ~ 0.05
    tr.add(starts[ok] + 0.15 * tx, np.full(ok.sum(), tx * link), link)  # download
    return tr, dict(period=period)


def fam_ps_diloco(rng):
    """Lopsided outer sync every H steps: long gaps, big bursts."""
    link = rng.choice([12.5e9, 50e9])
    tc = rng.uniform(0.2, 1.0)
    H = rng.choice([10, 30, 100])
    tx = rng.uniform(0.2, 2.0)
    period, jit = H * tc + tx, rng.uniform(0.01, 0.05)
    n = int(DUR / period) + 2
    starts = np.cumsum(period * (1 + jit * rng.standard_normal(n))) - period * rng.uniform()
    ok = (starts > 0) & (starts < DUR)
    tr = Trace()
    tr.add(starts[ok], np.full(ok.sum(), tx * link), link)
    return tr, dict(period=period)


def fam_ps_jitter(rng):
    """ADAPTIVE attacker: lopsided PS with random idle inserted before each step
    to disguise the cadence. Idle ~ Exponential(kappa * period): throughput falls
    to 1 / (1 + kappa)."""
    link = rng.choice([12.5e9, 50e9])
    tx, tc = rng.uniform(0.05, 1.0), rng.uniform(0.2, 2.0)
    kappa = rng.choice([0.25, 0.5, 1.0, 2.0])
    base = tc + tx
    gaps = base + rng.exponential(kappa * base, int(DUR / base) + 2)
    starts = np.cumsum(gaps) - base * rng.uniform()
    ok = (starts > 0) & (starts < DUR)
    tr = Trace()
    tr.add(starts[ok] + 0.15 * tx, np.full(ok.sum(), tx * link), link)
    return tr, dict(period=base, kappa=kappa, slowdown=1 + kappa)


def fam_pp_serve(rng):
    """Two-stage pipeline serving with continuous batching: a decode handoff
    every decode step, sized by the number of active sequences; a prefill
    handoff per arriving request."""
    link = rng.choice([12.5e9, 50e9])
    hidden = rng.choice([4096, 8192])
    td = rng.uniform(0.015, 0.08)                # decode step, s
    lam = rng.uniform(0.5, 20)                   # requests/s
    arr = poisson_times(lam, rng)
    out_len = rng.lognormal(np.log(256), 0.8, len(arr)).clip(8, 2048)
    ends = arr + out_len * td
    steps = np.arange(0, DUR, td)
    steps = steps + rng.normal(0, 0.1 * td, len(steps))
    active = np.searchsorted(arr, steps) - np.searchsorted(np.sort(ends), steps)
    tr = Trace()
    tr.add(steps.clip(0, DUR - 1e-3), np.maximum(active, 0) * hidden * 2.0, link)
    prompt = rng.lognormal(np.log(1200), 0.9, len(arr)).clip(128, 16000)
    tr.add(arr, prompt * hidden * 2.0, link)
    return tr, dict(lam=lam)


def fam_disagg(rng):
    """Disaggregated serving: each request's KV cache pushed prefill -> decode."""
    link = rng.choice([12.5e9, 50e9])
    lam = rng.uniform(0.2, 10)
    kv = rng.choice([128e3, 320e3])              # bytes per token
    arr = poisson_times(lam, rng)
    prompt = rng.lognormal(np.log(1200), 0.9, len(arr)).clip(128, 16000)
    tr = Trace()
    tr.add(arr, prompt * kv, link)
    return tr, dict(lam=lam)


def fam_bursty(rng):
    """Pipeline serving under on/off load (traffic spikes)."""
    link = rng.choice([12.5e9, 50e9])
    hidden = rng.choice([4096, 8192])
    on, off = rng.uniform(5, 30), rng.uniform(5, 30)
    lam_on = rng.uniform(5, 30)
    t, arr = 0.0, []
    while t < DUR:
        a = t + poisson_times(lam_on, rng)
        arr.append(a[a < t + on])
        t += on + off
    arr = np.concatenate(arr)
    arr = arr[arr < DUR]
    prompt = rng.lognormal(np.log(1200), 0.9, len(arr)).clip(128, 16000)
    tr = Trace()
    tr.add(arr, prompt * hidden * 2.0, link)
    tr.add(arr + rng.uniform(0, 1, len(arr)), prompt * hidden * 0.5, link)
    return tr, dict(lam=lam_on)


def fam_static_batch(rng):
    """Offline batch inference with static batches: prefill a batch, decode it to
    completion, repeat. Genuinely periodic serving -- the hard negative."""
    link = rng.choice([12.5e9, 50e9])
    hidden = rng.choice([4096, 8192])
    B = rng.choice([16, 32, 64])
    td = rng.uniform(0.015, 0.08)
    L = rng.choice([64, 128, 256, 512])
    period = L * td + rng.uniform(0.2, 1.0)
    tr = Trace()
    t = rng.uniform(0, period)
    while t < DUR:
        tr.add(np.array([t]), np.array([B * rng.uniform(500, 2000) * hidden * 2.0]), link)
        steps = t + 0.2 + np.arange(L) * td
        steps = steps[steps < DUR]
        tr.add(steps, np.full(len(steps), B * hidden * 2.0), link)
        t += period * (1 + 0.02 * rng.standard_normal())
    return tr, dict(period=period)


def fam_ps_paced(rng):
    """FULLY ADAPTIVE attacker: lopsided PS whose dense download is cut into many
    small pieces of random size, sent at random times spread across the whole
    step, with random idle between steps. Pacing is free while the link has
    headroom; the idle costs throughput 1 / (1 + kappa)."""
    link = rng.choice([12.5e9, 50e9])
    tx, tc = rng.uniform(0.05, 1.0), rng.uniform(0.2, 2.0)
    kappa = rng.choice([0.0, 0.25, 0.5, 1.0])
    base = tc + tx
    gaps = base + rng.exponential(kappa * base, int(DUR / base) + 2) if kappa else \
        np.full(int(DUR / base) + 2, base)
    starts = np.cumsum(gaps) - base * rng.uniform()
    tr = Trace()
    M = tx * link
    for s0, g in zip(starts, gaps):
        if s0 >= DUR or s0 + g <= 0:
            continue
        k = rng.integers(50, 400)
        sizes = rng.lognormal(0, 1.0, k); sizes *= M / sizes.sum()
        times = np.sort(s0 + rng.uniform(0, g, k))
        ok = (times > 0) & (times < DUR)
        tr.add(times[ok], sizes[ok], link)
    return tr, dict(period=base, kappa=kappa, slowdown=1 + kappa)


def fam_ps_stealth(rng):
    """STEALTH attacker: DiLoCo-style sync every H steps, outer update compressed
    ~100x, trickled out continuously in random pieces over the whole round. Volume
    is then ~model_bytes / (100 * H * step) -- at or below serving's. Whether a
    model trains well under that much compression is an open ML question; the
    traffic is what this models."""
    link = rng.choice([12.5e9, 50e9])
    model_bytes = rng.choice([2.7e9, 13.4e9])
    H, tc = rng.choice([50, 100, 500]), rng.uniform(0.2, 2.0)
    comp = rng.choice([30, 100, 300])
    rate = model_bytes / comp / (H * tc)                   # bytes/s, averaged
    n = int(rate * DUR / rng.uniform(2e5, 5e6)) + 20        # piece count
    times = np.sort(rng.uniform(0, DUR, n))
    sizes = rng.lognormal(0, 1.0, n); sizes *= rate * DUR / sizes.sum()
    tr = Trace()
    tr.add(times, sizes, link)
    return tr, dict(period=H * tc, rate=rate)


FAMILIES = {
    "ps_step": (1, fam_ps_step), "ps_diloco": (1, fam_ps_diloco),
    "ps_jitter": (1, fam_ps_jitter), "ps_paced": (1, fam_ps_paced), "ps_stealth": (1, fam_ps_stealth),
    "pp_serve": (0, fam_pp_serve), "disagg": (0, fam_disagg),
    "bursty_serve": (0, fam_bursty), "static_batch": (0, fam_static_batch),
}


# ── observation ─────────────────────────────────────────────────────────────
def observe(tr: Trace, hz: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Counter read at `hz` with 1% timing jitter -> rate per sample interval."""
    cum = np.cumsum(tr.b)
    n = int(DUR * hz)
    t = np.arange(n) / hz + rng.normal(0, 0.01 / hz, n)
    t = np.maximum.accumulate(np.clip(t, 0, DUR))
    c = cum[np.clip((t / DT).astype(int), 0, len(cum) - 1)]
    return t[1:], np.diff(c) / np.maximum(np.diff(t), 1e-9)


def _acf_peak(x):
    x = x - x.mean()
    if x.std() < 1e-12 or len(x) < 8:
        return 0.0
    f = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[:len(x)]
    ac = ac / ac[0]
    lo = 2
    return float(ac[lo:len(x) // 2].max()) if len(x) // 2 > lo else 0.0


def window_features(t, r) -> list[dict]:
    rows, w = [], 0.0
    while w + WIN <= DUR + 1e-9:
        m = (t >= w) & (t < w + WIN)
        x = r[m]
        if len(x) >= 4 and x.max() > 0:
            thr = 0.5 * np.percentile(x, 99)
            on = x > thr
            onsets = np.flatnonzero(on[1:] & ~on[:-1]) + 1
            iv = np.diff(t[m][onsets]) if len(onsets) >= 3 else np.array([])
            y = x - x.mean()
            p = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
            p[0] = 0
            rows.append(dict(
                duty=float((x > 0.05 * np.percentile(x, 99)).mean()),
                cv=float(x.std() / x.mean()),
                acf_peak=_acf_peak(x),
                fft_peak_frac=float(p.max() / p.sum()) if p.sum() > 0 else 0.0,
                onset_cv=float(iv.std() / iv.mean()) if len(iv) >= 2 and iv.mean() > 0 else 1.0,
                log_onsets=float(np.log1p(len(onsets))),
                log_rate=float(np.log10(1 + x.mean()))))
        w += STRIDE
    return rows


FEATS = ["duty", "cv", "acf_peak", "fft_peak_frac", "onset_cv", "log_onsets"]
VOL = ["log_rate"]


def build(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for fam, (y, fn) in FAMILIES.items():
        for j in range(N_PER_FAMILY):
            tr, meta = fn(rng)
            for hz in RATES:
                t, r = observe(tr, hz, rng)
                for i, f in enumerate(window_features(t, r)):
                    rows.append(dict(family=fam, job=f"{fam}_{j}", is_train=y, hz=hz, w=i,
                                     **f, **{k: v for k, v in meta.items()}))
    return pd.DataFrame(rows)


FEATURE_SETS = {
    "all": FEATS,
    "interval regularity only": ["onset_cv", "log_onsets"],
    "autocorrelation only": ["acf_peak"],
    "all but autocorrelation": [f for f in FEATS if f != "acf_peak"],
}


ATTACKERS = {"ps_jitter", "ps_paced", "ps_stealth"}


def lofo(df: pd.DataFrame, feats=FEATS) -> pd.DataFrame:
    """Per sample rate. Ordinary families (every family except the adaptive
    attackers): leave one out, fit a logistic regression on the rest, score the
    held-out one. Adaptive attackers: never trained on -- scored by a model fitted
    on every ordinary family, as a defender would meet them. Reported: pooled AUC
    over the ordinary families, and each family's share of windows called
    training."""
    out = []
    base = [f for f in FAMILIES if f not in ATTACKERS]
    for hz, d in df.groupby("hz"):
        sc = []
        for fam in base:
            tr, te = d[d.family.isin(base) & (d.family != fam)], d[d.family == fam]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(tr[feats].fillna(1.0), tr.is_train)
            sc.append(te.assign(p=clf.predict_proba(te[feats].fillna(1.0))[:, 1]))
        full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        tr = d[d.family.isin(base)]
        full.fit(tr[feats].fillna(1.0), tr.is_train)
        for fam in sorted(ATTACKERS):
            te = d[d.family == fam]
            sc.append(te.assign(p=full.predict_proba(te[feats].fillna(1.0))[:, 1]))
        sc = pd.concat(sc)
        b = sc[sc.family.isin(base)]
        row = dict(hz=hz, auc=roc_auc_score(b.is_train, b.p))
        for fam in FAMILIES:
            row[fam] = float((sc[sc.family == fam].p > 0.5).mean())
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = build()
    df.to_parquet(OUT / "windows.parquet", index=False)
    pd.set_option("display.width", 220)
    allres = []
    for name, feats in FEATURE_SETS.items():
        res = lofo(df, feats).assign(features=name)
        allres.append(res)
        print(f"\n== {name} ==")
        print(res.drop(columns="features").round(3).to_string(index=False))
    pd.concat(allres).to_csv(OUT / "lofo_by_rate.csv", index=False)
