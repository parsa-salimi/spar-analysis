"""Per-GPU features: Robi Rahman's detector inputs, computed with his code.

`sliding_windows` from his branch's classifier/threeway_improved.py, 30 s
windows at a 15 s stride, causal run-level features -- his deployed setting.

Streams: within a job, GPUs of the same node run the same program and their
windows are near-duplicates, so four streams per job are windowed (GPUs 0 and 4
on each node) rather than sixteen. Each stream is scored separately and the job's
per-window score is the median across streams.
"""
from __future__ import annotations
import pandas as pd
from . import data as D
from .jobs import robi

GPUS = (0, 4)
WIN, STRIDE = 30.0, 15.0


def job_gpu_windows(files: pd.DataFrame, cell: str) -> pd.DataFrame:
    T, _ = robi()
    sel = files[files.gpu.isin(GPUS) & files.available]
    frames = []
    for r in sel.itertuples():
        d = T._normalize_columns(D.read(r))
        d["run_id"] = f"{r.rid}_n{r.node}g{r.gpu}"
        d["workload_label"] = cell
        d["threeway_label"] = T.threeway_label(cell)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    w = T.sliding_windows(df, WIN, STRIDE, run_level_mode="causal")
    if not w.empty:
        w["stream"] = w.run_id
        w["w_abs"] = w.window_start
    return w
