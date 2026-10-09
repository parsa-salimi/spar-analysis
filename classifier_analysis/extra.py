"""Three further experiments for the interconnect-importance report.

  family     leave-one-family-out with deliberately small feature sets
             (symmetry only, the balance group, balance + volume) and simple
             models (a one-split tree, logistic regression) next to the full
             sets. Question: do learned models fail on unseen families because
             learning fails, or because they latch on to the wrong features?
  adversarial  models fitted on every honest window, then scored on the
             adversarial jobs, per feature set. Thresholds chosen by grouped
             (by workload) CV on the honest data only.
  curve      learning curve: hold out one workload, train on m of the others
             (random draws, both classes present), score the held-out one.
             Question: does more training coverage close the gap between
             feature sets?

    python3 -m classifier_analysis.extra family|adversarial|curve
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from . import common as C
from network_detector import rule as R

OUT = C.RES / "extra"


def rf(n=200, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    return lambda: RandomForestClassifier(n_estimators=n, min_samples_leaf=2, max_features="sqrt",
                                          class_weight="balanced", random_state=seed, n_jobs=2)


def logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def stump():
    return lambda: DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=0)


def sets(gc):
    bal = [c for c in C.NIC if C.group_of(c) == "NIC balance"]
    vol = [c for c in C.NIC if C.group_of(c) == "NIC volume"]
    return {"symmetry only": ["sym"], "NIC balance (5)": bal, "NIC balance + volume": bal + vol,
            "NIC (14)": C.NIC, "GPU (all 174)": gc, "Everything": gc + C.NIC}


# ── 1. family split, restricted features ────────────────────────────────────
def family():
    D, gc = C.load()
    y = D.is_train.values
    rows, preds = [], []
    for fs, cols in sets(gc).items():
        Xa = C.X(D, cols)
        for mname, mk in (("one-split tree", stump()), ("Logistic", logistic()), ("RandomForest", rf())):
            P = []
            for fold, tr, te in C.splits(D, "family", 0):
                p, m = C.fit_predict(mk, Xa[tr], y[tr], Xa[te])
                P.append(D.iloc[te][["cell", "rid", "w", "family", "is_train"]].assign(p=p))
            P = pd.concat(P)
            mt = C.metrics(P)
            rows.append(dict(fs=fs, model=mname, **mt))
            preds.append(P.assign(fs=fs, model=mname))
            print(f"{fs:22s} {mname:15s} TPR {mt['TPR']:.3f} FPR {mt['FPR']:.3f} AUC {mt['AUC']:.3f}", flush=True)
    _, call = R.call(D)
    rows.append(dict(fs="symmetry rule (unfitted)", model="rule", **C.metrics(D.assign(p=call.astype(float).values))))
    # what threshold does a one-split tree on symmetry learn, fold by fold?
    th = []
    for fold, tr, te in C.splits(D, "family", 0):
        t = DecisionTreeClassifier(max_depth=1, class_weight="balanced").fit(D.iloc[tr][["sym"]], y[tr])
        th.append(dict(held_out=D.iloc[te].family.iloc[0], feature="sym", threshold=float(t.tree_.threshold[0])))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "family_restricted.csv", index=False)
    pd.DataFrame(th).to_csv(OUT / "family_stump_thresholds.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "family_restricted_preds.parquet", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(th).round(3).to_string(index=False))


# ── 2. adversarial jobs ─────────────────────────────────────────────────────
def best_thr(y, p):
    best, bt = -1, 0.5
    for t in np.unique(np.concatenate([[0.5], np.quantile(p, np.linspace(0.01, 0.99, 99))])):
        c = p >= t
        ba = (c[y == 1].mean() + 1 - c[y == 0].mean()) / 2
        if ba > best:
            best, bt = ba, t
    return bt


def adversarial():
    import xgboost as xgb
    A, gc = C.load(include_adversarial=True)
    H, Adv = A[A.kind == "honest"].reset_index(drop=True), A[A.kind == "adversarial"].reset_index(drop=True)
    y = H.is_train.values
    fsd = {"NIC (14)": C.NIC, "GPU (all 174)": gc,
           "GPU, no interconnect": [c for c in gc if not c.startswith("pcie")], "Everything": gc + C.NIC}
    mk = {"RandomForest": rf(),
          "XGBoost": lambda: xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                               random_state=0, n_jobs=2, verbosity=0)}
    out = []
    for fs, cols in fsd.items():
        Xh, Xa = C.X(H, cols), C.X(Adv, cols)
        for mname, m in mk.items():
            oof = np.zeros(len(H))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xh, y, H.cell.values):
                oof[te], _ = C.fit_predict(m, Xh[tr], y[tr], Xh[te])
            thr = best_thr(y, oof)
            p, _ = C.fit_predict(m, Xh, y, Xa)
            out.append(Adv[["cell", "rid", "w"]].assign(p=p, thr=thr, call=p >= thr, fs=fs, model=mname))
            print(fs, mname, f"thr={thr:.3f}", flush=True)
    _, call = R.call(Adv)
    out.append(Adv[["cell", "rid", "w"]].assign(p=call.astype(float).values, thr=0.5,
                                               call=call.values, fs="symmetry rule", model="rule"))
    P = pd.concat(out)
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_parquet(OUT / "adversarial_preds.parquet", index=False)
    T = P.groupby(["cell", "fs", "model"]).call.mean().unstack(["fs", "model"])
    T.to_csv(OUT / "adversarial_by_featureset.csv")
    pd.set_option("display.width", 250)
    print(T.round(2).to_string())


# ── 3. learning curve ───────────────────────────────────────────────────────
def curve():
    D, gc = C.load()
    fsd = {"NIC (14)": C.NIC, "GPU (all 174)": gc, "Everything": gc + C.NIC}
    cells = D.groupby("cell").is_train.first()
    pos, neg = cells[cells == 1].index.tolist(), cells[cells == 0].index.tolist()
    rng = np.random.RandomState(0)
    M, DRAWS = [3, 6, 10, 15, 20], 3
    Xs = {fs: C.X(D, cols) for fs, cols in fsd.items()}
    y = D.is_train.values
    rows, t0 = [], time.time()
    for held in cells.index:
        te = np.where(D.cell.values == held)[0]
        others_p = [c for c in pos if c != held]; others_n = [c for c in neg if c != held]
        for m in M:
            for d in range(DRAWS if m < 20 else 1):
                if m >= len(others_p) + len(others_n):
                    pick = others_p + others_n
                else:
                    k_p = max(1, round(m * len(pos) / len(cells)))
                    k_p = min(k_p, len(others_p), m - 1)
                    pick = list(rng.choice(others_p, k_p, replace=False)) + \
                           list(rng.choice(others_n, m - k_p, replace=False))
                tr = np.where(D.cell.isin(pick).values)[0]
                for fs in fsd:
                    p, _ = C.fit_predict(rf(150, d), Xs[fs][tr], y[tr], Xs[fs][te])
                    rows.append(dict(held=held, is_train=int(cells[held]), m=len(pick), draw=d, fs=fs,
                                     called=float((p >= 0.5).mean()), p_mean=float(p.mean())))
        print(f"held {held:40s} {time.time()-t0:5.0f}s", flush=True)
    R_ = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R_.to_csv(OUT / "learning_curve_raw.csv", index=False)
    g = R_.groupby(["fs", "m", "draw", "is_train"]).called.mean().unstack("is_train")
    g["BalAcc"] = (g[1] + 1 - g[0]) / 2
    S = g.groupby(["fs", "m"]).agg(TPR=(1, "mean"), FPR=(0, "mean"), BalAcc=("BalAcc", "mean"),
                                   BalAcc_sd=("BalAcc", "std")).reset_index()
    S.to_csv(OUT / "learning_curve.csv", index=False)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    {"family": family, "adversarial": adversarial, "curve": curve}[sys.argv[1]]()
