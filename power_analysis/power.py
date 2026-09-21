"""Repeated held-out-GROUP evaluation at five levels of statistical grouping.

For each level we repeatedly (a) partition the GROUPS into train/val/test,
(b) fit a classifier on the training windows, (c) record window-level accuracy
on val and test plus the per-group test accuracies.

Point of the exercise: accuracy is not one number, it is a DISTRIBUTION whose
width is set by the number of independent groups in the test split, not by the
number of windows. Everything downstream (error bars, model selection,
required-n) follows from that distribution.

Usage:  python3 power.py <level> <n_reps> <seed_base>
Appends to reps_<level>.csv and groups_<level>.csv, so it can be run in
several passes and the results pooled.
"""
from __future__ import annotations
import sys, re, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config  # noqa: F401  (resolves paths + puts GPU-monitoring on sys.path)
from config import RESULTS as PA, DATA
from classifier.threeway_improved import get_feature_cols
from families import fine, family, coarse

JOB_RE = re.compile(r"_([0-9a-f]{6,16})_(\d{8})_(\d{6})$")

# At most this many evenly-spaced windows per run. Windows inside a run are
# near-duplicates (ICC ~ 0.98), so the cap costs almost no information and makes
# the repeated-split sweep tractable. sensitivity.py shows the conclusion does
# not depend on it: 3, 10 and 30 give the same spread.
WINDOW_CAP = 10


def _cap(g, k):
    return g if len(g) <= k else g.iloc[np.linspace(0, len(g) - 1, k).astype(int)]


def build(cap: int = WINDOW_CAP):
    """Feature matrix, labels and grouping columns, derived from the committed
    window cache. Cached to data/ on first call."""
    cache = DATA / f"features_cap{cap}.npz"
    meta_p = DATA / f"meta_cap{cap}.parquet"
    if cache.exists() and meta_p.exists():
        z = np.load(cache); X, y = z["X"], z["y"]
        meta = pd.read_parquet(meta_p)
    else:
        W = (pd.read_parquet(DATA / "windows_dc3.parquet")
               .sort_values(["run_id", "window_start"]))
        Wc = W.groupby("run_id", group_keys=False).apply(_cap, cap)
        feats = get_feature_cols(Wc)
        X = Wc[feats].fillna(0).values.astype(np.float32)
        y = (Wc.threeway_label.values == "ml_training").astype(int)
        meta = Wc[["run_id", "workload_label", "threeway_label",
                   "gpu_name", "window_start"]].reset_index(drop=True)
        np.savez_compressed(cache, X=X, y=y)
        meta.to_parquet(meta_p, index=False)
    base = meta.workload_label.astype(str).str.replace(r"_gpu\d+", "", regex=True)

    def job_of(stem, b):
        m = JOB_RE.search(str(stem))
        return f"{b}|{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else f"{b}|{stem}"

    meta = meta.assign(
        g_run=meta.run_id.astype(str),
        g_job=[job_of(s, b) for s, b in zip(meta.run_id, base)],
        g_fine=meta.workload_label.map(fine),
        g_family=meta.workload_label.map(family),
        g_coarse=meta.workload_label.map(coarse),
    )
    return meta, X, y


def one_rep(X, y, g, rng, frac=(0.6, 0.2, 0.2)):
    """One random group-level train/val/test partition. Returns None if the
    draw cannot give every split both classes (rare at coarse levels)."""
    gu = np.array(sorted(pd.unique(g)))
    # a group is single-class by construction (label is a function of workload)
    gcls = pd.Series(y).groupby(g).max().reindex(gu).values
    for _ in range(60):
        perm = rng.permutation(len(gu))
        n1 = int(round(frac[0]*len(gu))); n2 = n1 + int(round(frac[1]*len(gu)))
        idx_tr, idx_va, idx_te = perm[:n1], perm[n1:n2], perm[n2:]
        ok = all(len(np.unique(gcls[i])) == 2 and len(i) >= 2
                 for i in (idx_tr, idx_va, idx_te))
        if ok:
            break
    else:
        return None
    sel = lambda idx: np.isin(g, gu[idx])
    m_tr, m_va, m_te = sel(idx_tr), sel(idx_va), sel(idx_te)
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=int(rng.integers(1 << 30)), n_jobs=-1)
    clf.fit(X[m_tr], y[m_tr])
    correct_va = clf.predict(X[m_va]) == y[m_va]
    correct_te = clf.predict(X[m_te]) == y[m_te]
    per_group = pd.DataFrame({"grp": g[m_te], "correct": correct_te}) \
                  .groupby("grp")["correct"].agg(["mean", "size"])
    return dict(
        acc_val=float(correct_va.mean()), acc_test=float(correct_te.mean()),
        acc_test_groupmean=float(per_group["mean"].mean()),
        n_grp_train=len(idx_tr), n_grp_val=len(idx_va), n_grp_test=len(idx_te),
        n_win_train=int(m_tr.sum()), n_win_test=int(m_te.sum()),
    ), per_group


def main():
    level, n_reps, seed_base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    meta, X, y = build()
    g = meta[f"g_{level}"].values
    print(f"level={level}  groups={len(pd.unique(g))}  windows={len(y)}", flush=True)
    reps, glong, t0 = [], [], time.time()
    for r in range(n_reps):
        rng = np.random.default_rng(seed_base + r)
        out = one_rep(X, y, g, rng)
        if out is None:
            continue
        rec, per_group = out
        rec.update(level=level, rep=seed_base + r)
        reps.append(rec)
        pg = per_group.reset_index().rename(columns={"mean": "acc", "size": "n_win"})
        pg["rep"] = seed_base + r; pg["level"] = level
        glong.append(pg)
        if r % 20 == 0:
            print(f"  {r}/{n_reps}  {time.time()-t0:.0f}s", flush=True)
    for df, name in [(pd.DataFrame(reps), f"reps_{level}.csv"),
                     (pd.concat(glong, ignore_index=True), f"groups_{level}.csv")]:
        p = PA/name
        df.to_csv(p, mode="a", header=not p.exists(), index=False)
    a = pd.DataFrame(reps).acc_test
    print(f"DONE {level}: n={len(a)} mean={a.mean():.4f} sd={a.std(ddof=1):.4f} "
          f"min={a.min():.4f} max={a.max():.4f}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
