"""Rule plus GPU telemetry: three ways to combine them, scored on unseen families.

The October corpus breaks the network rule in two known directions:
  false alarms  balanced communication that is not gradient averaging
                (DLRM all-to-all, sharded FFT, stencil halo exchange, NCCL sweep)
  misses        training whose cross-node traffic is one-way or absent
                (IMPALA / Ape-X actor-learner RL, SWARM, long-interval LocalSGD)
GPU telemetry is the obvious second opinion for both. Designs, each scored by
leave-one-family-out over honest families (models never see the held-out family):

  rule                 the frozen rule alone
  GPU                  RandomForest on Robi's 174 GPU features, all windows
  Everything           RandomForest on GPU + 14 network features
  rule AND GPU         training only if the rule fires and the GPU model agrees
  rule OR GPU          training if either fires
  rule -> stage 2      a GPU model trained ONLY on windows the rule fires on
                       (so it learns "which balanced traffic is training"),
                       applied to the windows the rule fires on
  stage 2 OR GPU-quiet the stage-2 verdict where the rule fires; elsewhere the
                       all-window GPU model, so silent training can still be caught

Job-level: majority of windows, and any window.

    python3 -m classifier_analysis.combined
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from . import common as C
from network_detector import rule as R

OUT = C.RES / "combined"


def rf(seed=0):
    return lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt",
                                          class_weight="balanced", random_state=seed, n_jobs=2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D, gc = C.load()
    y = D.is_train.values
    _, call = R.call(D)
    D["rule"] = call.values.astype(float)
    Xg, Xe = C.X(D, gc), C.X(D, gc + C.NIC)
    pos = D.rule.values == 1
    p_g, p_e, p_s2 = np.full(len(D), np.nan), np.full(len(D), np.nan), np.full(len(D), np.nan)
    t0 = time.time()
    for f in sorted(D.family.unique()):
        te = D.family.values == f
        tr = ~te
        p_g[te], _ = C.fit_predict(rf(), Xg[tr], y[tr], Xg[te])
        p_e[te], _ = C.fit_predict(rf(), Xe[tr], y[tr], Xe[te])
        trs, tes = tr & pos, te & pos
        if tes.any() and len(set(y[trs])) == 2:
            p_s2[tes], _ = C.fit_predict(rf(), Xg[trs], y[trs], Xg[tes])
        print(f"{f:24s} {time.time()-t0:5.0f}s", flush=True)
    D["p_gpu"], D["p_every"], D["p_s2"] = p_g, p_e, p_s2
    D[["cell", "rid", "w", "family", "is_train", "rule", "p_gpu", "p_every", "p_s2"]].to_parquet(
        OUT / "combined_preds.parquet", index=False)
    g, e, s2 = D.p_gpu >= 0.5, D.p_every >= 0.5, (D.p_s2 >= 0.5).fillna(False)
    r = D.rule == 1
    designs = {"rule": r, "GPU": g, "Everything": e, "rule AND GPU": r & g, "rule OR GPU": r | g,
               "rule -> stage 2": r & s2, "stage 2 OR GPU-quiet": (r & s2) | (~r & g)}
    rows, per = [], []
    for k, c in designs.items():
        m = C.metrics(D.assign(p=c.astype(float).values))
        jb = D.assign(c=c.values).groupby(["rid", "is_train"]).c.agg(["mean", "max"]).reset_index()
        m["TPR_job_any"] = float(jb[jb.is_train == 1]["max"].mean())
        m["FPR_job_any"] = float(jb[jb.is_train == 0]["max"].mean())
        rows.append(dict(design=k, **{a: round(b, 3) for a, b in m.items()}))
        per.append(D.assign(c=c.values).groupby(["is_train", "family"]).c.mean().rename(k))
    T = pd.DataFrame(rows); T.to_csv(OUT / "combined.csv", index=False)
    F = pd.concat(per, axis=1).round(2); F.to_csv(OUT / "combined_by_family.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(T.to_string(index=False)); print(F.to_string())


if __name__ == "__main__":
    main()
