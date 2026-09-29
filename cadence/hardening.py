#!/usr/bin/env python3
"""Can a classifier be hardened against every attack? Three regimes, on the
simulated traffic of cadence.py (see README there for what is and is not modelled).

  unseen      train on ordinary families only; every attack is new to it
  hardened    train on ordinary families AND every other attack; the held-out
              family -- attack or not -- is new (leave one family out over all)
  memorised   attacks in training, tested on new runs of the SAME attack types
              (5-fold grouped by job): the best case for hardening

Feature sets: shape (scale-free) and shape + volume (log mean byte rate).
Model: gradient-boosted trees, so that "hardening" is not limited by a linear
boundary. Reported per family: share of windows called training.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import cadence as C

df = C.build()
df.to_parquet(C.OUT / "windows_v2.parquet", index=False)
FS = {"shape": C.FEATS, "shape+volume": C.FEATS + C.VOL}
base = [f for f in C.FAMILIES if f not in C.ATTACKERS]
fams = list(C.FAMILIES)


def model():
    return HistGradientBoostingClassifier(max_iter=200, random_state=0)


rows = []
for hz in (10, 50):
    d = df[df.hz == hz]
    for fs, cols in FS.items():
        X = lambda z: z[cols].fillna(1.0)
        # unseen
        sc = []
        for f in base:
            tr = d[d.family.isin(base) & (d.family != f)]
            sc.append(d[d.family == f].assign(p=model().fit(X(tr), tr.is_train).predict_proba(X(d[d.family == f]))[:, 1]))
        m = model().fit(X(d[d.family.isin(base)]), d[d.family.isin(base)].is_train)
        for f in C.ATTACKERS:
            sc.append(d[d.family == f].assign(p=m.predict_proba(X(d[d.family == f]))[:, 1]))
        u = pd.concat(sc)
        # hardened: leave one family out over ALL families
        sc = []
        for f in fams:
            tr = d[d.family != f]
            sc.append(d[d.family == f].assign(p=model().fit(X(tr), tr.is_train).predict_proba(X(d[d.family == f]))[:, 1]))
        h = pd.concat(sc)
        # memorised: grouped k-fold by job over everything
        sc = []
        for tri, tei in GroupKFold(5).split(d, d.is_train, d.job):
            tr, te = d.iloc[tri], d.iloc[tei]
            sc.append(te.assign(p=model().fit(X(tr), tr.is_train).predict_proba(X(te))[:, 1]))
        mm = pd.concat(sc)
        for name, s in (("unseen", u), ("hardened", h), ("memorised", mm)):
            r = dict(hz=hz, features=fs, regime=name)
            for f in fams:
                r[f] = float((s[s.family == f].p > 0.5).mean())
            rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(C.OUT / "hardening.csv", index=False)
pd.set_option("display.width", 250)
order = ["hz", "features", "regime"] + [f for f in fams if f in C.ATTACKERS or C.FAMILIES[f][0] == 1] + \
        [f for f in fams if C.FAMILIES[f][0] == 0]
print(R[order].round(2).to_string(index=False))
