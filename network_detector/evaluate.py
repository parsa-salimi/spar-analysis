"""Network-only vs per-GPU vs combined detection, on the same two-node jobs.

Protocol (fixed before any result was looked at)
  * Honest jobs only in training. Leave-one-family-out over the honest families:
    fit on every other honest family, score the held-out one.
  * One model for every arm: RandomForest, 200 trees, default settings -- the
    configuration Robi Rahman's family-holdout analysis found best, with no
    selection on validation.
  * Unit: a job's 30 s window. The per-GPU arm scores each GPU stream and takes
    the median across the job's streams in that window.
  * Called training: score > 0.5. Reported per held-out family: the share of its
    windows called training (TPR for training families, FPR for the rest), plus
    pooled AUC over every held-out window.
  * Adversarial jobs are never trained on. They are scored by a model fitted on
    all honest families.

Arms
  net        per-node network features (every job has them)
  net_novol  the same without byte volume, because volume on this cluster
             reflects one TCP gVNIC per node rather than a production fabric
  gpu        Robi's per-GPU window features, his code
  both       gpu + net
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from . import jobs as J

HERE = Path(__file__).resolve().parent
NET = ["vol", "sym", "dir_abs", "duty", "cv", "acf1", "acf3", "fft_peak_frac",
       "burst_sym", "peak_mean"]
NET_NOVOL = [c for c in NET if c != "vol"]


def load():
    T, _ = J.robi()
    NW = pd.read_parquet(HERE / "data/net_windows.parquet")
    GW = pd.read_parquet(HERE / "data/gpu_windows.parquet")
    jt = J.table()[["cell", "rid", "label", "is_train", "family", "kind"]]
    NW = NW.merge(jt, on=["cell", "rid"], how="inner")
    GW = GW.merge(jt, on=["cell", "rid"], how="inner", suffixes=("", "_j"))
    gcols = [c for c in T.get_feature_cols(GW) if c not in
             ("w", "w_abs", "is_train", "stream") and not c.endswith("_j")]
    GW = GW.merge(NW[["cell", "rid", "w"] + NET], on=["cell", "rid", "w"], how="inner")
    return NW, GW, gcols


def rf():
    return RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=0)


def _X(df, cols):
    return np.nan_to_num(df[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def score(arm, train, test, NW, GW, gcols):
    """Returns per job-window scores for `test` families after fitting on `train`."""
    if arm in ("net", "net_novol"):
        cols = NET if arm == "net" else NET_NOVOL
        tr, te = NW[NW.family.isin(train) & (NW.kind == "honest")], NW[NW.family.isin(test)]
        m = rf().fit(_X(tr, cols), tr.is_train)
        return te.assign(p=m.predict_proba(_X(te, cols))[:, 1])[
            ["cell", "rid", "w", "family", "kind", "is_train", "p"]]
    cols = gcols if arm == "gpu" else gcols + NET
    tr, te = GW[GW.family.isin(train) & (GW.kind == "honest")], GW[GW.family.isin(test)]
    m = rf().fit(_X(tr, cols), tr.is_train)
    te = te.assign(p=m.predict_proba(_X(te, cols))[:, 1])
    return (te.groupby(["cell", "rid", "w", "family", "kind", "is_train"], as_index=False)
              .p.median())



from . import rule as R

ARMS = ["net", "gpu", "both", "rule", "combined"]
IDLE_UTIL, IDLE_MEM_MB = 5.0, 2048.0


def screen(NW, GW):
    """Drop honest jobs in which nothing ran: every sampled GPU below IDLE_UTIL %
    utilisation with under IDLE_MEM_MB loaded. Such jobs carry their cell's label
    but are idle hardware (e.g. 2 of 3 s16_train_fsdp7b_seq2k reps). Adversarial
    jobs are kept: some are idle by design (oracle-gated cells that never train)."""
    j = GW.groupby(["cell", "rid", "kind"]).agg(
        util=("gpu_utilization_pct_mean", "median"), mem=("mem_used_mb_mean", "median")).reset_index()
    idle = j[(j.kind == "honest") & (j.util < IDLE_UTIL) & (j.mem < IDLE_MEM_MB)]
    bad = set(zip(idle.cell, idle.rid))
    keep = lambda d: d[[(c, r) not in bad for c, r in zip(d.cell, d.rid)]]
    return keep(NW), keep(GW), idle[["cell", "rid"]]


def decide(arm, train, test, NW, GW, gcols):
    """Per job-window decision (called training or not) for families `test`."""
    base = NW[NW.family.isin(test)][["cell", "rid", "w", "family", "kind", "is_train",
                                     "vol", "sym"]]
    if arm in ("rule", "combined"):
        act, tr = R.call(base)
        base = base.assign(active=act.values, rule=tr.values)
    if arm == "rule":
        return base.assign(call=base.rule)
    s = score("gpu" if arm == "combined" else arm, train, test, NW, GW, gcols)
    m = base.merge(s[["cell", "rid", "w", "p"]], on=["cell", "rid", "w"], how="inner")
    if arm == "combined":
        return m.assign(call=np.where(m.active, m.rule, m.p > 0.5))
    return m.assign(call=m.p > 0.5)


def main():
    NW, GW, gcols = load()
    NW, GW, dropped = screen(NW, GW)
    both_w = GW[["cell", "rid", "w"]].drop_duplicates()
    NW = NW.merge(both_w, on=["cell", "rid", "w"], how="inner")   # same windows, every arm
    honest = sorted(NW[NW.kind == "honest"].family.unique())
    advf = sorted(NW[NW.kind == "adversarial"].family.unique())
    out = HERE / "results"; out.mkdir(exist_ok=True)
    per, adv = [], []
    for a in ARMS:
        for f in honest:
            d = decide(a, [x for x in honest if x != f], [f], NW, GW, gcols)
            d = d[d.kind == "honest"]
            per.append(d.assign(arm=a)[["arm", "family", "cell", "rid", "w", "is_train", "call"]])
        d = decide(a, honest, advf, NW, GW, gcols)
        adv.append(d[d.kind == "adversarial"].assign(arm=a)[["arm", "family", "cell", "rid", "w", "call"]])
    P, A = pd.concat(per), pd.concat(adv)
    fam = P.groupby(["is_train", "family", "arm"]).call.mean().unstack("arm")[ARMS]
    fam["jobs"] = P[P.arm == "rule"].groupby(["is_train", "family"]).rid.nunique()
    fam["windows"] = P[P.arm == "rule"].groupby(["is_train", "family"]).size()
    tot = P.groupby(["arm", "is_train"]).call.mean().unstack("is_train").reindex(ARMS)
    tot.columns = ["FPR (window)", "TPR (window)"]
    job = (P.groupby(["arm", "rid", "is_train"]).call.mean().gt(0.5)
            .groupby(["arm", "is_train"]).mean().unstack("is_train").reindex(ARMS))
    tot["FPR (job)"], tot["TPR (job)"] = job[0], job[1]
    advc = A.groupby(["cell", "arm"]).call.mean().unstack("arm")[ARMS]
    # sensitivity of the rule to its two thresholds
    h = NW[NW.kind == "honest"]; rate = 10 ** h.vol - 1
    sens = pd.DataFrame([dict(active_mibps=a_, sym_min=s_,
                              TPR=float(((rate >= a_) & (h.sym >= s_))[h.is_train == 1].mean()),
                              FPR=float(((rate >= a_) & (h.sym >= s_))[h.is_train == 0].mean()))
                         for a_ in (0.1, 0.3, 1, 3, 10) for s_ in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)])
    fam.to_csv(out / "lofo_per_family.csv"); tot.to_csv(out / "totals.csv")
    advc.to_csv(out / "adversarial.csv"); sens.to_csv(out / "rule_sensitivity.csv", index=False)
    dropped.to_csv(out / "idle_jobs_dropped.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"idle honest jobs dropped: {len(dropped)}  ->  {dropped.cell.value_counts().to_dict()}")
    print("\nheld-out family, share of windows called training:")
    print(fam.round(3).to_string())
    print("\ntotals over honest held-out windows:"); print(tot.round(3).to_string())
    print("\nadversarial cells (fit on all honest families), share called training:")
    print(advc.round(3).to_string())
    print(f"\nrule sensitivity: TPR {sens.TPR.min():.3f}-{sens.TPR.max():.3f}, "
          f"FPR {sens.FPR.min():.3f}-{sens.FPR.max():.3f} over the 30-point grid")


if __name__ == "__main__":
    main()
