"""Check: does reducing GPU features to a job-window median change the GPU arm?

Robi's own scoring trains and scores each GPU stream separately and takes the
median probability per job-window (network_detector/evaluate.py does the same).
Here both GPU-only and Everything are rerun that way under the workload and
family splits, to compare with the median-of-features rows in ablation_*.csv.

    python3 -m classifier_analysis.perstream
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from . import common as C
from .robust import models


def main():
    gc = C.gpu_cols()
    N = pd.read_parquet(C.DATA / "net.parquet")
    G = pd.read_parquet(C.DATA / "gpu.parquet")
    D, _ = C.load()                       # the job-windows used everywhere else
    keys = D[["cell", "rid", "w"]]
    G = G.drop(columns=[c for c in C.NIC if c in G.columns])
    S = G.merge(keys, on=["cell", "rid", "w"]).merge(
        N[["cell", "rid", "w"] + C.NIC], on=["cell", "rid", "w"])
    S[C.NIC_CROSS] = S[C.NIC_CROSS].fillna(0.0)
    y = S.is_train.values
    rows = []
    for fs, cols in {"GPU (all 174)": gc, "Everything": gc + C.NIC}.items():
        Xa = C.X(S, cols)
        for proto, seeds in (("workload", [0, 1, 2]), ("family", [0])):
            for mname in ("RandomForest", "XGBoost"):
                for seed in seeds:
                    mk = models(seed)[mname]
                    if proto == "family":
                        fam = sorted(S.family.unique())
                        sp = [(np.where(S.family != f)[0], np.where(S.family == f)[0]) for f in fam]
                    else:
                        sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                                  .split(Xa, y, S.cell.values))
                    preds = []
                    for tr, te in sp:
                        p, _ = C.fit_predict(mk, Xa[tr], y[tr], Xa[te])
                        preds.append(S.iloc[te][["cell", "rid", "w", "is_train"]].assign(p=p))
                    P = pd.concat(preds).groupby(["cell", "rid", "w", "is_train"], as_index=False).p.median()
                    m = C.metrics(P)
                    rows.append(dict(fs=fs, protocol=proto, model=mname, seed=seed, **m))
                print(fs, proto, mname, flush=True)
    R = pd.DataFrame(rows).groupby(["fs", "protocol", "model"]).mean(numeric_only=True).drop(columns="seed")
    R.to_csv(C.RES / "perstream_check.csv")
    pd.set_option("display.width", 200)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
