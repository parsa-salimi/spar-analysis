"""Which telemetry channel carries the training signal?

Fits every (feature set x model) under three hold-out protocols and keeps every
held-out prediction, so all metrics are computed afterwards from the same pool.

  protocols   run (paper headline), workload, family   -- see common.splits
  seeds       3 for run/workload (fold assignment), 1 for family (deterministic)
  feature     common.feature_sets: NIC, NIC without volume, PCIe, NIC + PCIe,
  sets        GPU without interconnect, GPU (all 174), Everything
  models      RandomForest, XGBoost (the paper's configurations), Logistic
  reference   the frozen network rule (fitted to nothing), on the same windows

Honest jobs only; adversarial jobs are never trained on.

    python3 -m classifier_analysis.ablation          # resumable; ~30 min on 2 cores
"""
from __future__ import annotations
import itertools, sys, time
import numpy as np
import pandas as pd
from . import common as C
from network_detector import rule as R

SEEDS = {"run": [0], "workload": [0, 1, 2], "family": [0]}
KEEP = ["cell", "rid", "w", "family", "is_train"]


def main():
    D, gc = C.load()
    FS = C.feature_sets(gc)
    out = C.RES / "preds"; out.mkdir(parents=True, exist_ok=True)
    y = D.is_train.values
    t0 = time.time()
    # reference: the rule, identical for every protocol (nothing is fitted)
    _, call = R.call(D)
    D[KEEP].assign(p=call.astype(float).values).to_parquet(out / "rule.parquet", index=False)
    mk = C.models()
    jobs = list(itertools.product(SEEDS, FS, mk))
    for k, (proto, fs, mname) in enumerate(jobs):
        f = out / f"{proto}__{fs}__{mname}.parquet".replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "and")
        if f.exists():
            continue
        cols, rows = FS[fs], []
        Xall = C.X(D, cols)
        for seed in SEEDS[proto]:
            for fold, tr, te in C.splits(D, proto, seed):
                p, _ = C.fit_predict(C.models(seed)[mname], Xall[tr], y[tr], Xall[te])
                rows.append(D.iloc[te][KEEP].assign(p=p, seed=seed, fold=fold))
        pd.concat(rows).assign(protocol=proto, fs=fs, model=mname).to_parquet(f, index=False)
        print(f"[{k+1}/{len(jobs)}] {proto:8s} {fs:28s} {mname:12s} {time.time()-t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()
