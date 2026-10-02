"""Export the labelled window tables and the per-second NIC series that the
classifier analysis needs, so it can run anywhere without the GPU-monitoring
clone or its LFS objects.

    python3 -m classifier_analysis.export      # from spar-analysis/

Writes classifier_analysis/data/:
  net.parquet      network windows (network_detector.build), with label/family/kind
  gpu.parquet      per-GPU windows (Robi's features), same labels, same windows
  nic_1hz.parquet  per-second tx/rx per node for every kept two-node job
  jobs.parquet     the job table (cell, rid, label, family, kind)
"""
from pathlib import Path
import pandas as pd
from network_detector import evaluate as E, data as D, jobs as J, build as B

OUT = Path(__file__).resolve().parent / "data"


def main():
    OUT.mkdir(exist_ok=True)
    NW, GW, gcols = E.load()
    NW, GW, dropped = E.screen(NW, GW)
    NW.to_parquet(OUT / "net.parquet", index=False)
    GW.to_parquet(OUT / "gpu.parquet", index=False)
    pd.Series(gcols, name="col").to_frame().to_parquet(OUT / "gpu_cols.parquet", index=False)
    dropped.to_parquet(OUT / "idle_dropped.parquet", index=False)
    jt = J.table()
    jt[["cell", "rid", "label", "is_train", "family", "kind", "scope"]].to_parquet(OUT / "jobs.parquet", index=False)
    keep = set(zip(NW.cell, NW.rid))
    ix = D.index()
    ix = ix[[(c, r) in keep for c, r in zip(ix.cell, ix.rid)]]
    nic, _ = B.dedup(ix)
    rows = []
    for r in nic.itertuples():
        d = D.read(r)[["timestamp_epoch", "nic_tx_mbps", "nic_rx_mbps"]].copy()
        d["cell"], d["rid"], d["node"] = r.cell, r.rid, int(r.node)
        rows.append(d)
    S = pd.concat(rows, ignore_index=True)
    S.to_parquet(OUT / "nic_1hz.parquet", index=False)
    print(f"net {len(NW)} rows / {NW.rid.nunique()} jobs; gpu {len(GW)} rows, {len(gcols)} gpu cols; "
          f"nic_1hz {len(S)} rows / {S.rid.nunique()} jobs; dropped idle {len(dropped)}")


if __name__ == "__main__":
    main()
