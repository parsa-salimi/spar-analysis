"""Robustness checks for the workload-held-out comparison.

1. Threshold transfer vs ranking. A held-out workload can be ranked correctly
   (high AUC) yet fall on the wrong side of 0.5. So for each outer fold an inner
   grouped-by-cell CV on the training set picks the threshold that maximises
   balanced accuracy, and that threshold is applied to the held-out fold. Also
   reported: AUC and TPR at 1% / 5% FPR (threshold set on the test pool, the same
   way for every feature set).
2. Drop-group retraining. Does "Everything" depend on its network features?
   Retrain without each NIC group, and GPU plus one NIC group at a time.
3. Per-stream GPU scoring, Robi's way (score each GPU stream, median of the
   probabilities per job-window), against the median-of-features used elsewhere.

Workload split (StratifiedGroupKFold by cell, 5 folds), seeds 0-2,
RandomForest (200 trees) and XGBoost (paper config).

    python3 -m classifier_analysis.robust
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from . import common as C

SEEDS = [0, 1]


def models(seed):
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=2),
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
            n_jobs=2, verbosity=0),
    }


def best_threshold(y, p):
    grid = np.unique(np.concatenate([[0.5], np.quantile(p, np.linspace(0.01, 0.99, 99))]))
    best, bt = -1, 0.5
    for t in grid:
        c = p >= t
        if (y == 1).any() and (y == 0).any():
            ba = (c[y == 1].mean() + 1 - c[y == 0].mean()) / 2
            if ba > best:
                best, bt = ba, t
    return bt


def feature_sets(gc):
    g = lambda name: [c for c in C.NIC if C.group_of(c) == name]
    bal, tim, vol = g("NIC balance"), g("NIC timing"), g("NIC volume")
    return {
        "NIC": C.NIC,
        "GPU (all 174)": gc,
        "GPU, no interconnect": [c for c in gc if not c.startswith("pcie")],
        "Everything": gc + C.NIC,
        "Everything - NIC balance": gc + tim + vol,
        "Everything - NIC volume": gc + bal + tim,
        "GPU + NIC balance only": gc + bal,
        "GPU + NIC volume only": gc + vol,
        "GPU + NIC timing only": gc + tim,
        "NIC balance only": bal,
    }


def main():
    D, gc = C.load()
    y = D.is_train.values
    FS = feature_sets(gc)
    out = C.RES / "robust"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for fs, cols in FS.items():
        for mname in ("RandomForest", "XGBoost"):
            f = out / f"{fs}__{mname}.parquet".replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            if f.exists():
                continue
            Xa = C.X(D, cols)
            rows = []
            for seed in SEEDS:
                mk = models(seed)[mname]
                cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                for fold, (tr, te) in enumerate(cv.split(Xa, y, D.cell.values)):
                    # inner CV on the training cells -> threshold
                    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
                    oof = np.zeros(len(tr))
                    for itr, ite in inner.split(Xa[tr], y[tr], D.cell.values[tr]):
                        p, _ = C.fit_predict(mk, Xa[tr][itr], y[tr][itr], Xa[tr][ite])
                        oof[ite] = p
                    thr = best_threshold(y[tr], oof)
                    p, _ = C.fit_predict(mk, Xa[tr], y[tr], Xa[te])
                    rows.append(D.iloc[te][["cell", "rid", "w", "family", "is_train"]]
                                .assign(p=p, thr=thr, seed=seed, fold=fold))
            pd.concat(rows).assign(fs=fs, model=mname).to_parquet(f, index=False)
            print(f"{fs:28s} {mname:12s} {time.time()-t0:6.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
