"""Feature importance: which signals the learned detectors actually use.

Three views, because single-feature importances mislead when features are
correlated (and 174 NVML statistics are highly correlated):

  impurity     RandomForest mean decrease in impurity (fast, biased toward
               high-cardinality features) -- reported for context only.
  permutation  drop in balanced accuracy on held-out data when one feature's
               column is shuffled. The honest measure of "does the model lean
               on this", computed on the held-out fold.
  grouped      permutation importance when a whole physical group is shuffled
               together (common.group_of): PCIe, NIC balance, Power & thermal,
               Memory, Compute, ... This is the number that answers "how much
               does the interconnect channel matter", unconfounded by having
               split one source into many columns.

Run under the paper's run-grouped split and under family hold-out, on the
"Everything" feature set (GPU 174 + NIC 14), RandomForest.

    python3 -m classifier_analysis.importance
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from . import common as C


def perm_importance(m, Xte, yte, cols, groups, rng, n=3):
    base = balanced_accuracy_score(yte, m.predict(Xte))
    single, grouped = {}, {}
    for j, c in enumerate(cols):
        drops = []
        for _ in range(n):
            Xp = Xte.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - balanced_accuracy_score(yte, m.predict(Xp)))
        single[c] = float(np.mean(drops))
    for g in sorted(set(groups)):
        idx = [j for j, c in enumerate(cols) if C.group_of(c) == g]
        if not idx:
            continue
        drops = []
        for _ in range(n):
            Xp = Xte.copy()
            for j in idx:
                Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - balanced_accuracy_score(yte, m.predict(Xp)))
        grouped[g] = float(np.mean(drops))
    return base, single, grouped


def run(D, cols, protocol, seed=0):
    import time
    from sklearn.ensemble import RandomForestClassifier
    y = D.is_train.values
    Xall = C.X(D, cols)
    groups = [C.group_of(c) for c in cols]
    imp = np.zeros(len(cols)); single = {c: [] for c in cols}; grouped = {}
    rng = np.random.RandomState(seed)
    nfold = 0
    folds = list(C.splits(D, protocol, seed))
    for fold, tr, te in folds:
        if len(set(y[te])) < 2:
            continue
        t = time.time()
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=seed, n_jobs=2).fit(Xall[tr], y[tr])
        imp += m.feature_importances_
        _, s, g = perm_importance(m, Xall[te], y[te], cols, groups, rng)
        for c in cols:
            single[c].append(s[c])
        for k, v in g.items():
            grouped.setdefault(k, []).append(v)
        nfold += 1
        print(f"  {protocol} fold {nfold}/{len(folds)}  {time.time()-t:.0f}s", flush=True)
    impurity = pd.Series(imp / nfold, index=cols, name="impurity")
    perm = pd.Series({c: np.mean(v) for c, v in single.items()}, name="perm")
    grp = pd.Series({k: np.mean(v) for k, v in grouped.items()}, name="perm_drop_balacc")
    return impurity, perm, grp.sort_values(ascending=False)


def main():
    D, gc = C.load()
    cols = gc + C.NIC
    out = C.RES; out.mkdir(parents=True, exist_ok=True)
    for proto in ("run", "workload"):
        impurity, perm, grp = run(D, cols, proto)
        top = pd.concat([impurity, perm], axis=1).sort_values("perm", ascending=False)
        top.head(30).to_csv(out / f"importance_single_{proto}.csv")
        grp.to_csv(out / f"importance_grouped_{proto}.csv")
        print(f"\n=== {proto} : grouped permutation importance (drop in balanced accuracy) ===")
        print(grp.round(4).to_string())
        print(f"\n--- {proto} : top 15 single features (permutation) ---")
        print(top.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
