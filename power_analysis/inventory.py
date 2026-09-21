import sys, json, warnings
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
warnings.filterwarnings("ignore")
import config  # noqa: F401  (resolves paths + puts GPU-monitoring on sys.path)
from config import GPU_ROOT as ROOT, RESULTS, DATA, FIGURES
from classifier.threeway_improved import threeway_label, _normalize_gpu_name
rows=[]
paths = sorted(ROOT.rglob("*.parquet"))
for i,p in enumerate(paths):
    try:
        f = pq.ParquetFile(p)
        cols = set(f.schema_arrow.names)
        n = f.metadata.num_rows
        want = [c for c in ("workload_label","gpu_name","run_id","world_size","num_gpus","hostname","node_rank") if c in cols]
        wl=gpu=rid=None; extra={}
        if want:
            t = f.read(columns=want).to_pandas()
            if "workload_label" in t: wl = str(t["workload_label"].iloc[0])
            if "gpu_name" in t: gpu = _normalize_gpu_name(t["gpu_name"].iloc[0])
            if "run_id" in t: rid = str(t["run_id"].iloc[0])
            for c in ("world_size","num_gpus","hostname","node_rank"):
                if c in t: extra[c]=str(t[c].iloc[0])
        stem=p.stem
        if wl is None:
            wl = stem.split("_NVIDIA_")[0] if "_NVIDIA_" in stem else stem.rsplit("_",3)[0]
        if rid is None: rid = stem
        rows.append(dict(path=str(p.relative_to(ROOT)), top=p.relative_to(ROOT).parts[0],
                         stem=stem, n_rows=n, workload_label=wl, gpu_name=gpu, run_id=rid,
                         threeway=threeway_label(wl), ncols=len(cols),
                         has_nvlink=int(any("nvlink" in c.lower() for c in cols)),
                         schema_key="|".join(sorted(cols))[:400], **extra))
    except Exception as e:
        rows.append(dict(path=str(p.relative_to(ROOT)), top=p.relative_to(ROOT).parts[0], stem=p.stem,
                         n_rows=-1, workload_label="ERR", error=str(e)[:120]))
    if i%200==0: print(i, len(paths), flush=True)
df=pd.DataFrame(rows)
df.to_csv(RESULTS/"inventory.csv", index=False)
print("DONE", len(df))
