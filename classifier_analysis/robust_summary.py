"""Summarise results/robust/: threshold-free metrics, tuned thresholds, drop-group
retraining, and cell-level bootstrap intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from . import common as C


def tpr_at_fpr(y, p, fpr):
    neg = np.sort(p[y == 0])[::-1]
    k = int(np.floor(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((p[y == 1] > thr).mean())


def per_seed(g):
    y, p = g.is_train.values, g.p.values
    c05, ct = p >= 0.5, p >= g.thr.values
    ba = lambda c: (c[y == 1].mean() + 1 - c[y == 0].mean()) / 2
    return dict(BalAcc_05=ba(c05), BalAcc_tuned=ba(ct),
                TPR_tuned=ct[y == 1].mean(),
                FPR_tuned=ct[y == 0].mean(),
                AUC=roc_auc_score(y, p),
                TPR_at_1pctFPR=tpr_at_fpr(y, p, 0.01),
                TPR_at_5pctFPR=tpr_at_fpr(y, p, 0.05))


def cell_bootstrap(P, n=2000, seed=0):
    """Balanced accuracy (tuned threshold) with cells resampled within class.
    Uses each window's call averaged over seeds."""
    P = P.assign(c=(P.p >= P.thr).astype(float))
    w = P.groupby(["cell", "rid", "w", "is_train"]).c.mean().reset_index()
    cells = {k: g for k, g in w.groupby("cell")}
    pos = [c for c, g in cells.items() if g.is_train.iloc[0] == 1]
    neg = [c for c, g in cells.items() if g.is_train.iloc[0] == 0]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        cp = rng.choice(pos, len(pos)); cn = rng.choice(neg, len(neg))
        tp = np.concatenate([cells[c].c.values for c in cp]).mean()
        fp = np.concatenate([cells[c].c.values for c in cn]).mean()
        out.append((tp + 1 - fp) / 2)
    return np.percentile(out, [2.5, 97.5])


def main():
    d = C.RES / "robust"
    P = pd.concat([pd.read_parquet(f) for f in d.glob("*.parquet")], ignore_index=True)
    rows = []
    for (fs, m), g in P.groupby(["fs", "model"]):
        s = pd.DataFrame([per_seed(x) for _, x in g.groupby("seed")]).mean()
        lo, hi = cell_bootstrap(g)
        rows.append(dict(fs=fs, model=m, **s.round(3).to_dict(), BalAcc_tuned_CI=f"[{lo:.2f}, {hi:.2f}]"))
    T = pd.DataFrame(rows).sort_values(["model", "BalAcc_tuned"], ascending=[True, False])
    T.to_csv(C.RES / "robust_workload.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
