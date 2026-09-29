"""Build the joined window tables for every two-node (s16) job with NIC logs.

Output, in network_detector/data/:
  net_windows.parquet   one row per (job, window): network features
  gpu_windows.parquet   one row per (GPU stream, window): Robi's per-GPU features,
                        tagged with the network window of the same job and time

A job is (cell, NIC run id). GPU streams are matched to it by run id; for
replicated inference, where every replica has its own run id and none matches
the NIC log, by cell and overlapping time instead. A GPU window is joined to the
network window whose start is within half a stride of its own.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, pandas as pd
from . import data as D, netfeat as N, gpufeat as G, jobs as J

OUT = Path(__file__).resolve().parent / "data"


def dedup(ix):
    ix = ix.sort_values("bytes", ascending=False)
    nic = ix[ix.kind == "nic"].drop_duplicates(["cell", "rid", "node"])
    gpu = ix[ix.kind == "gpu"].drop_duplicates(["cell", "rid", "node", "gpu"])
    return nic, gpu


def _span(r):
    d = D.read(r)
    t = d["timestamp_epoch"] if "timestamp_epoch" in d else pd.to_datetime(d.timestamp_utc).astype("int64") / 1e9
    return float(t.min()), float(t.max())


def main(budget_s=165):
    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    jt = J.table()
    keep = set(zip(jt.cell, jt.rid))
    ix = D.index()
    ix = ix[[(c, r) in keep for c, r in zip(ix.cell, ix.rid)] & ix.cell.str.startswith("s16_")]
    nic, gpu = dedup(ix)
    part = D.CACHE / "build_parts"; part.mkdir(exist_ok=True)
    todo = list(nic.groupby(["cell", "rid"]))
    done = 0
    for (cell, rid), g in todo:
        pf = part / f"{cell}__{rid}.pkl"
        if pf.exists():
            done += 1; continue
        if time.time() - t_start > budget_s:
            break
        nodes = {int(r.node): D.read(r) for r in g.itertuples()}
        if 0 not in nodes:
            nodes = {0: nodes[min(nodes)]}
        nw = N.job_windows(nodes)
        gf = gpu[(gpu.cell == cell) & (gpu.rid == rid)]
        if gf.empty:                       # replicated inference: match by time
            t0 = min(d.timestamp_epoch.min() for d in nodes.values())
            t1 = max(d.timestamp_epoch.max() for d in nodes.values())
            cand = gpu[(gpu.cell == cell) & gpu.gpu.isin(G.GPUS)]
            ok = []
            for r in cand.itertuples():
                a, b = _span(r)
                if a < t1 and b > t0:
                    ok.append(r.Index)
            gf = cand.loc[ok]
        gw = G.job_gpu_windows(gf, cell) if not gf.empty else pd.DataFrame()
        pd.to_pickle((nw.assign(cell=cell, rid=rid), gw.assign(cell=cell, rid=rid)
                      if not gw.empty else gw), pf)
        done += 1
    print(f"{done}/{len(todo)} jobs built  ({time.time()-t_start:.0f}s)")
    if done < len(todo):
        return False
    nws, gws = [], []
    for pf in sorted(part.glob("*.pkl")):
        nw, gw = pd.read_pickle(pf)
        if nw.empty:
            continue
        nw = nw.reset_index(drop=True); nw["w"] = np.arange(len(nw))
        nws.append(nw)
        if not gw.empty:
            idx = np.searchsorted(nw.w_abs.values, gw.w_abs.values)
            best = []
            for i, t in zip(idx, gw.w_abs.values):
                c = [k for k in (i - 1, i) if 0 <= k < len(nw)]
                k = min(c, key=lambda k: abs(nw.w_abs.values[k] - t))
                best.append(k if abs(nw.w_abs.values[k] - t) <= N.STRIDE / 2 else -1)
            gw = gw.assign(w=best)
            gws.append(gw[gw.w >= 0])
    NW = pd.concat(nws, ignore_index=True)
    GW = pd.concat(gws, ignore_index=True)
    for c in GW.columns:
        if GW[c].dtype == np.float64 and c not in ("w_abs", "window_start", "window_end"):
            GW[c] = GW[c].astype(np.float32)
    NW.to_parquet(OUT / "net_windows.parquet", index=False)
    GW.to_parquet(OUT / "gpu_windows.parquet", index=False)
    print(f"net windows {len(NW):,} over {NW.rid.nunique()} jobs; "
          f"gpu windows {len(GW):,} over {GW.stream.nunique()} streams, "
          f"{GW.groupby(['cell','rid']).ngroups} jobs")
    return True


if __name__ == "__main__":
    main()
