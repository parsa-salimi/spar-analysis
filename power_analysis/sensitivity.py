"""Sensitivity of the family-level result to the per-run window cap.

The main analysis keeps at most 10 evenly-spaced windows per run, on the
grounds that windows within a run are near-duplicates (ICC ~ 0.98). If that
cap were doing the work, the conclusion would be an artefact of it.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import config  # noqa: F401
from config import RESULTS as PA, DATA
from sklearn.ensemble import RandomForestClassifier
from classifier.threeway_improved import get_feature_cols
from families import family
from power import one_rep

W = pd.read_parquet(DATA/"windows_dc3.parquet").sort_values(["run_id","window_start"])
feats = get_feature_cols(W)
cap_k = int(sys.argv[1]); n_reps = int(sys.argv[2])
Wc = W.groupby("run_id", group_keys=False).apply(
    lambda g: g if len(g) <= cap_k else g.iloc[np.linspace(0, len(g)-1, cap_k).astype(int)])
X = Wc[feats].fillna(0).values.astype(np.float32)
y = (Wc.threeway_label.values == "ml_training").astype(int)
g = Wc.workload_label.map(family).values
accs, t0 = [], time.time()
for r in range(n_reps):
    out = one_rep(X, y, g, np.random.default_rng(90000+r))
    if out: accs.append(out[0]["acc_test"])
a = np.array(accs)
print(f"cap={cap_k:3d}  windows={len(X):6d}  reps={len(a):3d}  "
      f"mean={a.mean():.4f}  sd={a.std(ddof=1):.4f}  [{a.min():.3f},{a.max():.3f}]  "
      f"{time.time()-t0:.0f}s", flush=True)
pd.DataFrame(dict(cap=cap_k, acc=a)).to_csv(PA/"sensitivity.csv", mode="a",
    header=not (PA/"sensitivity.csv").exists(), index=False)
