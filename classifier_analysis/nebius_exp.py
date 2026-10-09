"""Classifier experiments on Long's Nebius InfiniBand corpus, and across corpora.

  within    Nebius only: job-grouped CV (3 folds over 18 jobs) and leave-one-
            workload-out (6 workloads).
  transfer  train on Robi's GCP corpus, test on Nebius, and the reverse.
            Cutoffs: 0.5, and a cutoff tuned by workload-grouped CV on the
            training corpus only.
  pooled    both corpora together, 10-fold grouped by workload (each workload
            held out once).

Feature sets: NIC (14 network features), NIC without volume, symmetry only,
GPU without interconnect, GPU all (incl. PCIe), Everything (GPU all + NIC), and
for Nebius-only runs NVLink (10) and Everything + NVLink. The symmetry rule is
the unfitted reference.

    python3 -m classifier_analysis.nebius_exp
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from . import common as C
from network_detector import rule as R

OUT = C.RES / "nebius"
KEEP = ["cell", "rid", "w", "family", "is_train", "corpus"]


def load():
    G, gc = C.load()
    G = G.assign(corpus="gcp")
    N = pd.read_parquet(C.DATA / "nebius_windows.parquet")
    for c in gc:
        if c not in N.columns:
            N[c] = 0.0
    N[C.NIC_CROSS] = N[C.NIC_CROSS].fillna(0.0)
    nvl = [c for c in N.columns if c.startswith("nvl_")]
    return G, N, gc, nvl


def models(seed=0):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                        max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=2),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        random_state=seed, n_jobs=2, verbosity=0),
        "Logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                        class_weight="balanced", max_iter=3000)),
        "one-split tree": lambda: DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                        random_state=0),
    }


def fsets(gc, nvl=None):
    s = {"NIC": C.NIC, "NIC without volume": [c for c in C.NIC if c != "vol"],
         "symmetry only": ["sym"],
         "GPU, no interconnect": [c for c in gc if not c.startswith("pcie")],
         "GPU, all": gc, "Everything": gc + C.NIC}
    if nvl:
        s["NVLink"] = nvl
        s["Everything + NVLink"] = gc + C.NIC + nvl
    return s


def best_thr(y, p):
    best, bt = -1, 0.5
    for t in np.unique(np.concatenate([[0.5], np.quantile(p, np.linspace(0.01, 0.99, 99))])):
        c = p >= t
        if (y == 1).any() and (y == 0).any():
            ba = (c[y == 1].mean() + 1 - c[y == 0].mean()) / 2
            if ba > best:
                best, bt = ba, t
    return bt


def tuned_threshold(make, X, y, cells, seed=0):
    oof = np.zeros(len(y))
    n = min(5, len(set(cells[y == 1])), len(set(cells[y == 0])))
    for tr, te in StratifiedGroupKFold(max(n, 2), shuffle=True, random_state=seed).split(X, y, cells):
        oof[te], _ = C.fit_predict(make, X[tr], y[tr], X[te])
    return best_thr(y, oof)


def summarise(P, by):
    rows = []
    for k, g in P.groupby(by):
        m = C.metrics(g)
        anyw = g.assign(c=g.p >= 0.5).groupby(["rid", "is_train"]).c.max().reset_index()
        m["TPR_job_any"] = float(anyw[anyw.is_train == 1].c.mean())
        m["FPR_job_any"] = float(anyw[anyw.is_train == 0].c.mean())
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), **m))
    return pd.DataFrame(rows)


def within(N, gc, nvl):
    y = N.is_train.values
    preds = []
    for fs, cols in fsets(gc, nvl).items():
        X = C.X(N, cols)
        for mname, mk in models().items():
            for proto in ("job-grouped", "workload held out"):
                if proto == "job-grouped":
                    sp = StratifiedGroupKFold(3, shuffle=True, random_state=0).split(X, y, N.rid.values)
                else:
                    sp = [(np.where(N.cell != c)[0], np.where(N.cell == c)[0]) for c in sorted(N.cell.unique())]
                for tr, te in sp:
                    p, _ = C.fit_predict(mk, X[tr], y[tr], X[te])
                    preds.append(N.iloc[te][KEEP].assign(p=p, fs=fs, model=mname, protocol=proto))
    _, call = R.call(N)
    for proto in ("job-grouped", "workload held out"):
        preds.append(N[KEEP].assign(p=call.astype(float).values, fs="symmetry rule", model="rule", protocol=proto))
    return pd.concat(preds)


def transfer(G, N, gc):
    preds = []
    for direction, A, B in (("GCP -> Nebius", G, N), ("Nebius -> GCP", N, G)):
        ya = A.is_train.values
        for fs, cols in fsets(gc).items():
            Xa, Xb = C.X(A, cols), C.X(B, cols)
            for mname, mk in models().items():
                thr = tuned_threshold(mk, Xa, ya, A.cell.values)
                p, _ = C.fit_predict(mk, Xa, ya, Xb)
                preds.append(B[KEEP].assign(p=p, thr=thr, fs=fs, model=mname, direction=direction))
        _, call = R.call(B)
        preds.append(B[KEEP].assign(p=call.astype(float).values, thr=0.5, fs="symmetry rule",
                                    model="rule", direction=direction))
        print(direction, "done", flush=True)
    return pd.concat(preds)


def pooled(G, N, gc):
    D = pd.concat([G, N], ignore_index=True)
    y = D.is_train.values
    preds = []
    for fs, cols in fsets(gc).items():
        X = C.X(D, cols)
        for mname in ("RandomForest", "XGBoost", "one-split tree"):
            mk = models()[mname]
            for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=0).split(X, y, D.cell.values):
                p, _ = C.fit_predict(mk, X[tr], y[tr], X[te])
                preds.append(D.iloc[te][KEEP].assign(p=p, fs=fs, model=mname))
        print("pooled", fs, flush=True)
    _, call = R.call(D)
    preds.append(D[KEEP].assign(p=call.astype(float).values, fs="symmetry rule", model="rule"))
    return pd.concat(preds)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G, N, gc, nvl = load()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    import os
    if not os.environ.get("SKIP_WITHIN"):
        W = within(N, gc, nvl)
        W.to_parquet(OUT / "within_preds.parquet", index=False)
        S = summarise(W, ["protocol", "fs", "model"]); S.to_csv(OUT / "within.csv", index=False)
        print(S.round(3).to_string(index=False), flush=True)
    T = transfer(G, N, gc)
    T.to_parquet(OUT / "transfer_preds.parquet", index=False)
    S = summarise(T, ["direction", "fs", "model"])
    St = summarise(T.assign(p=(T.p >= T.thr).astype(float)), ["direction", "fs", "model"])
    S = S.merge(St[["direction", "fs", "model", "TPR", "FPR", "BalAcc"]],
                on=["direction", "fs", "model"], suffixes=("", "_tuned"))
    S.to_csv(OUT / "transfer.csv", index=False)
    print(S.round(3).to_string(index=False), flush=True)
    P = pooled(G, N, gc)
    P.to_parquet(OUT / "pooled_preds.parquet", index=False)
    S = summarise(P, ["fs", "model"]); S.to_csv(OUT / "pooled.csv", index=False)
    print(S.round(3).to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
