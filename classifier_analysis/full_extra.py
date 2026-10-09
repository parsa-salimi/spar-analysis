"""Further experiments for the full report (GCP October corpus + Nebius PR #3).

  transfer   the detector designs (rule, GPU model, one model on both channels,
             rule -> stage 2, two-stage) fitted on ALL of Robi's honest GCP
             windows and scored, unchanged, on every Nebius job.
  pooled     both corpora together, leave-one-family-out over the union of
             families (shared names -- train_ddp, train_fsdp, train_lora,
             train_tp, train_pipe, train_moe, inferrep_qwen -- are held out
             together across both clusters), same designs.
  diversity  GCP only: how held-out-family accuracy grows with the number of
             families in the training set. Each draw holds out 8 random
             families (at least 2 training, 2 not) and trains on k of the
             other 34, k = 4, 8, 16, 34 (each subset has both classes).
  stage2     what the stage-2 GPU model looks at: impurity importance of a
             forest fitted on every GCP window the rule fires on, by feature
             and by physical group.

    python3 -m classifier_analysis.full_extra transfer|pooled|diversity|stage2
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from . import common as C
from network_detector import rule as R

OUT = C.RES / "full"
KEEP = ["cell", "rid", "w", "family", "is_train", "corpus"]


def rf(seed=0):
    return lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt",
                                          class_weight="balanced", random_state=seed, n_jobs=2)


def load_both():
    from .nebius_exp import load
    G, N, gc, _ = load()
    return G, N, gc


def designs(D):
    r = D.rule == 1
    g = D.p_gpu >= 0.5
    e = D.p_every >= 0.5
    s2 = (D.p_s2 >= 0.5).fillna(False)
    return {"rule": r, "GPU": g, "one model, both channels": e, "rule -> stage 2": r & s2,
            "two-stage": (r & s2) | (~r & g)}


def score(D, by=None):
    rows = []
    groups = [(None, D)] if by is None else list(D.groupby(by))
    for k, sub in groups:
        for name, c in designs(sub).items():
            m = C.metrics(sub.assign(p=c.astype(float).values))
            jb = sub.assign(c=c.values).groupby(["rid", "is_train"]).c.max().reset_index()
            m["TPR_job_any"] = float(jb[jb.is_train == 1].c.mean()) if (jb.is_train == 1).any() else np.nan
            m["FPR_job_any"] = float(jb[jb.is_train == 0].c.mean()) if (jb.is_train == 0).any() else np.nan
            rows.append(dict(**({} if by is None else {by: k}), design=name,
                             **{a: round(b, 3) for a, b in m.items()}))
    return pd.DataFrame(rows)


def fit_designs(Tr, Te, gc):
    """Fit GPU / both / stage-2 forests on Tr, return Te with p_gpu, p_every, p_s2."""
    y = Tr.is_train.values
    Te = Te.copy()
    Te["p_gpu"], _ = C.fit_predict(rf(), C.X(Tr, gc), y, C.X(Te, gc))
    Te["p_every"], _ = C.fit_predict(rf(), C.X(Tr, gc + C.NIC), y, C.X(Te, gc + C.NIC))
    pos_tr, pos_te = Tr.rule.values == 1, Te.rule.values == 1
    Te["p_s2"] = np.nan
    if pos_te.any() and len(set(y[pos_tr])) == 2:
        p, _ = C.fit_predict(rf(), C.X(Tr[pos_tr], gc), y[pos_tr], C.X(Te[pos_te], gc))
        Te.loc[pos_te, "p_s2"] = p
    return Te


def with_rule(D):
    _, call = R.call(D)
    return D.assign(rule=call.values.astype(int))


def transfer():
    G, N, gc = load_both()
    G, N = with_rule(G), with_rule(N)
    P = fit_designs(G, N, gc)
    P[KEEP + ["rule", "p_gpu", "p_every", "p_s2"]].to_parquet(OUT / "transfer_designs_preds.parquet", index=False)
    S = score(P); S.to_csv(OUT / "transfer_designs.csv", index=False)
    per = []
    for name, c in designs(P).items():
        per.append(P.assign(c=c.values).groupby(["is_train", "cell"]).c.mean().rename(name))
    F = pd.concat(per, axis=1).round(2); F.to_csv(OUT / "transfer_designs_by_cell.csv")
    print(S.to_string(index=False)); print(F.to_string())


def pooled():
    G, N, gc = load_both()
    D = with_rule(pd.concat([G, N], ignore_index=True))
    parts, t0 = [], time.time()
    for f in sorted(D.family.unique()):
        te = D.family.values == f
        if len(set(D.is_train[~te])) < 2:
            continue
        parts.append(fit_designs(D[~te], D[te], gc))
        print(f"{f:24s} {time.time()-t0:5.0f}s", flush=True)
    P = pd.concat(parts)
    P[KEEP + ["rule", "p_gpu", "p_every", "p_s2"]].to_parquet(OUT / "pooled_family_preds.parquet", index=False)
    S = pd.concat([score(P).assign(corpus="both"), score(P, "corpus")], ignore_index=True)
    S.to_csv(OUT / "pooled_family_designs.csv", index=False)
    per = []
    for name, c in designs(P).items():
        per.append(P.assign(c=c.values).groupby(["corpus", "is_train", "family"]).c.mean().rename(name))
    F = pd.concat(per, axis=1).round(2); F.to_csv(OUT / "pooled_family_by_family.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(S.to_string(index=False)); print(F.to_string())


def diversity(draws=8, ks=(4, 8, 16, 34)):
    D, gc = C.load()
    D = with_rule(D)
    fam = D.groupby("family").is_train.first()
    pos, neg = fam[fam == 1].index.tolist(), fam[fam == 0].index.tolist()
    sets = {"Network (14)": C.NIC, "GPU (174)": gc, "Both (188)": gc + C.NIC}
    Xs = {k: C.X(D, v) for k, v in sets.items()}
    y = D.is_train.values
    rng = np.random.RandomState(0)
    rows, t0 = [], time.time()
    for d in range(draws):
        test = list(rng.choice(pos, 3, replace=False)) + list(rng.choice(neg, 5, replace=False))
        rest_p = [f for f in pos if f not in test]; rest_n = [f for f in neg if f not in test]
        te = D.family.isin(test).values
        for k in ks:
            if k >= len(rest_p) + len(rest_n):
                pick = rest_p + rest_n
            else:
                kp = max(2, round(k * len(rest_p) / (len(rest_p) + len(rest_n))))
                pick = list(rng.choice(rest_p, kp, replace=False)) + list(rng.choice(rest_n, k - kp, replace=False))
            tr = D.family.isin(pick).values
            for name, X in Xs.items():
                p, _ = C.fit_predict(rf(d), X[tr], y[tr], X[te])
                m = C.metrics(D[te].assign(p=p))
                rows.append(dict(draw=d, k=len(pick), fs=name, **m))
            rows.append(dict(draw=d, k=len(pick), fs="rule", **C.metrics(D[te].assign(p=D.rule[te].astype(float).values))))
        print(f"draw {d} {time.time()-t0:5.0f}s", flush=True)
    R_ = pd.DataFrame(rows); R_.to_csv(OUT / "diversity_raw.csv", index=False)
    S = R_.groupby(["fs", "k"]).agg(BalAcc=("BalAcc", "mean"), BalAcc_sd=("BalAcc", "std"),
                                    AUC=("AUC", "mean"), TPR=("TPR", "mean"), FPR=("FPR", "mean")).round(3).reset_index()
    S.to_csv(OUT / "diversity.csv", index=False)
    print(S.to_string(index=False))


def stage2():
    D, gc = C.load()
    D = with_rule(D)
    P = D[D.rule == 1]
    m = rf()().fit(C.X(P, gc), P.is_train.values)
    imp = pd.Series(m.feature_importances_, index=gc).sort_values(ascending=False)
    grp = imp.groupby([C.group_of(c) for c in imp.index]).sum().sort_values(ascending=False)
    imp.head(20).round(4).to_csv(OUT / "stage2_top_features.csv")
    grp.round(4).to_csv(OUT / "stage2_groups.csv")
    neg = P[P.is_train == 0].groupby("family").size()
    print("rule-positive windows:", len(P), "training:", int(P.is_train.sum()), "not:", int((1 - P.is_train).sum()))
    print(neg.to_string()); print(grp.round(3).to_string()); print(imp.head(20).round(4).to_string())
    top = imp.head(6).index.tolist()
    print(P.groupby(["is_train", "family"])[top].median().round(2).to_string())
    P.groupby(["is_train", "family"])[top].median().round(3).to_csv(OUT / "stage2_top_feature_medians.csv")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"transfer": transfer, "pooled": pooled, "diversity": diversity, "stage2": stage2}[sys.argv[1]]()
