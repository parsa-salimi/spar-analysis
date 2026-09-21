import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import config  # noqa: F401  (resolves paths + puts GPU-monitoring on sys.path)
from config import GPU_ROOT as ROOT, RESULTS, DATA, FIGURES
from classifier.threeway_improved import (_normalize_columns, threeway_label, _normalize_gpu_name)
from fastwin import fast_causal_windows

WIN, STRIDE = 30.0, 15.0
inv = pd.read_csv(RESULTS/"inventory.csv")
DC3 = inv[inv.gpu_name.isin(["NVIDIA A100-SXM4-80GB","NVIDIA B200","NVIDIA H200"])]
import sys
A,B=int(sys.argv[1]),int(sys.argv[2])
DC3=DC3.iloc[A:B]
print("files",len(DC3),A,B,flush=True)

out, t0 = [], time.time()
for i, r in enumerate(DC3.itertuples()):
    p = ROOT/r.path
    try:
        df = pd.read_parquet(p)
        df = _normalize_columns(df)
        if "workload_label" not in df.columns:
            df["workload_label"] = r.workload_label
        df["run_id"] = r.stem                      # one file == one GPU stream
        if "gpu_name" not in df.columns: df["gpu_name"] = r.gpu_name
        df["gpu_name"] = df["gpu_name"].map(_normalize_gpu_name)
        df["threeway_label"] = df["workload_label"].apply(threeway_label)
        if "ts" not in df.columns or df["ts"].isna().all(): continue
        df = df.dropna(subset=["ts"]).sort_values("ts")
        w = fast_causal_windows(df, WIN, STRIDE)
        if len(w): out.append(w)
    except Exception as e:
        print("ERR", r.path, str(e)[:100], flush=True)
    if i % 50 == 0:
        print(f"{i}/{len(DC3)}  {time.time()-t0:.0f}s  windows={sum(len(x) for x in out)}", flush=True)

W = pd.concat(out, ignore_index=True)
KEEP64 = {"window_start", "window_end", "ts"}   # epoch seconds: float32 has ~128s
                                                # resolution at 1.78e9 and destroys them
for c in W.columns:
    if W[c].dtype == np.float64 and c not in KEEP64: W[c] = W[c].astype(np.float32)
W.to_parquet(DATA/f"win_{A}_{B}.parquet", index=False)
print("DONE windows", W.shape, f"{time.time()-t0:.0f}s", flush=True)
