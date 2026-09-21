"""Leave-one-family-out evaluation: every family is held out exactly once.

This is the estimator the corpus can actually support. Each family contributes
one near-independent observation, so n = (number of families) rather than
(number of windows), and the whole corpus is used for the point estimate
instead of a random 20% of it.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import config  # noqa: F401
from config import RESULTS as PA
from power import build
level = sys.argv[1] if len(sys.argv) > 1 else "family"
A, B = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10**9)
meta, X, y = build()
g = meta[f"g_{level}"].values
gu = np.array(sorted(pd.unique(g)))[A:B]
rows, t0 = [], time.time()
for i, f in enumerate(gu):
    te = g == f
    tr = ~te
    if len(np.unique(y[tr])) < 2:
        continue
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=0, n_jobs=-1).fit(X[tr], y[tr])
    corr = clf.predict(X[te]) == y[te]
    rows.append(dict(level=level, group=f, cls=int(y[te][0]), n_win=int(te.sum()),
                     acc=float(corr.mean())))
    if i % 15 == 0:
        print(f"  {i}/{len(gu)} {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
p = PA/f"lofo_{level}.csv"
df.to_csv(p, mode="a", header=not p.exists(), index=False)
print(f"DONE {len(df)} groups  window-acc={np.average(df.acc, weights=df.n_win):.4f}  "
      f"family-mean-acc={df.acc.mean():.4f}  {time.time()-t0:.0f}s")
