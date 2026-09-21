"""Fast causal sliding-window extraction.

Numerically identical to classifier.threeway_improved.sliding_windows with
run_level_mode="causal", but O(n) in run length instead of O(n^2): the causal
run-level features are recomputed from numpy prefixes and window chunks are cut
with searchsorted + iloc instead of boolean masks over the whole run.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import config  # noqa: F401  (puts GPU-monitoring on sys.path)
from classifier.threeway_improved import extract_enhanced_features


def _causal_run_feats(mem: np.ndarray, mem_total: float, upto: int) -> dict:
    m = mem[:upto]
    f = {}
    first_n = min(30, len(m))
    if first_n >= 5 and len(m) >= 5:
        mem_start = float(np.median(m[:3]))
        peak = float(np.max(m[:first_n]))
        d = peak - mem_start
        f["first_30s_mem_delta_mb"] = d
        f["first_30s_mem_delta_frac"] = d / max(mem_total, 1.0)
        f["first_30s_mem_ratio"] = peak / max(mem_start, 1.0)
    else:
        f["first_30s_mem_delta_mb"] = 0.0
        f["first_30s_mem_delta_frac"] = 0.0
        f["first_30s_mem_ratio"] = 1.0
    if len(m) >= 10:
        plateau = float(np.median(m[-min(30, len(m)):]))
        mem0 = float(m[0]); span = plateau - mem0
        if abs(span) < 1e-6:
            f["time_to_mem_plateau_samples"] = 0.0
        else:
            thr = mem0 + 0.9 * span
            hits = np.where(m >= thr if span > 0 else m <= thr)[0]
            f["time_to_mem_plateau_samples"] = float(hits[0]) if len(hits) else float(len(m))
    else:
        f["time_to_mem_plateau_samples"] = 0.0
    return f


def fast_causal_windows(df: pd.DataFrame, window_sec=30.0, stride_sec=15.0) -> pd.DataFrame:
    out = []
    for run_id, run_df in df.groupby("run_id"):
        run_df = run_df.sort_values("ts").reset_index(drop=True)
        if len(run_df) < 5:
            continue
        ts = run_df["ts"].values.astype(float)
        mem = run_df["mem_used_mb"].fillna(0).values.astype(float)
        mem_total = float(run_df["mem_total_mb"].iloc[0]) if "mem_total_mb" in run_df.columns else 1.0
        t0, t1 = ts[0], ts[-1]
        label = run_df["threeway_label"].iloc[0]
        wl = run_df["workload_label"].iloc[0]
        gpu = run_df["gpu_name"].iloc[0] if "gpu_name" in run_df.columns else ""

        def emit(i0, i1, ws, we):
            if i1 - i0 < 5:
                return
            feats = extract_enhanced_features(run_df.iloc[i0:i1])
            feats.update(_causal_run_feats(mem, mem_total, i1))
            feats.update(run_id=run_id, threeway_label=label, workload_label=wl,
                         gpu_name=gpu, window_start=ws, window_end=we)
            out.append(feats)

        if t1 - t0 < window_sec:
            emit(0, len(run_df), t0, t1)
        else:
            ws = t0
            while ws + window_sec <= t1 + 1:
                we = ws + window_sec
                i0 = int(np.searchsorted(ts, ws, "left"))
                i1 = int(np.searchsorted(ts, we, "left"))
                emit(i0, i1, ws, we)
                ws += stride_sec
    return pd.DataFrame(out) if out else pd.DataFrame()
