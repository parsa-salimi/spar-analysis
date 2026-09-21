"""Turn the repeated-split runs into the numbers the report needs."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
import config  # noqa: F401
from config import RESULTS as PA
from power import build, JOB_RE   # same grouping construction
LEVELS = ["run", "job", "fine", "family", "coarse"]
meta, X, y = build()

# ── replay each split (seeds are recorded) to recover the trivial baseline ──
def replay_baseline(level, rep, frac=(0.6, 0.2, 0.2)):
    g = meta[f"g_{level}"].values
    gu = np.array(sorted(pd.unique(g)))
    gcls = pd.Series(y).groupby(g).max().reindex(gu).values
    rng = np.random.default_rng(rep)
    for _ in range(60):
        perm = rng.permutation(len(gu))
        n1 = int(round(frac[0]*len(gu))); n2 = n1 + int(round(frac[1]*len(gu)))
        i_tr, i_va, i_te = perm[:n1], perm[n1:n2], perm[n2:]
        if all(len(np.unique(gcls[i])) == 2 and len(i) >= 2 for i in (i_tr, i_va, i_te)):
            break
    else:
        return None
    m_tr, m_te = np.isin(g, gu[i_tr]), np.isin(g, gu[i_te])
    maj = int(round(y[m_tr].mean()))            # majority class of the TRAIN split
    return float((y[m_te] == maj).mean())

rows, pooled = [], {}
for lv in LEVELS:
    R = pd.read_csv(PA/f"reps_{lv}.csv")
    G = pd.read_csv(PA/f"groups_{lv}.csv")
    R["baseline"] = [replay_baseline(lv, int(r)) for r in R.rep]
    a = R.acc_test.values
    n_te = R.n_grp_test.median()
    p = a.mean()
    # binomial prediction if each test GROUP were one independent Bernoulli(p)
    sd_binom = np.sqrt(p*(1-p)/n_te)
    # Kish design effect from unequal group sizes within each test split
    deff = G.groupby("rep")["n_win"].apply(lambda w: 1 + (w.std(ddof=0)/w.mean())**2).mean()
    sd_binom_deff = sd_binom*np.sqrt(deff)
    rho, pv = st.spearmanr(R.acc_val, R.acc_test)
    # how bimodal are per-group accuracies?  (ICC-like: groups are all-or-nothing)
    ga = G.acc.values
    rows.append(dict(level=lv, n_groups=meta[f"g_{lv}"].nunique(), reps=len(R),
        mean=p, sd=a.std(ddof=1), q05=np.quantile(a,.05), q95=np.quantile(a,.95),
        lo=a.min(), hi=a.max(), n_grp_test=n_te,
        baseline=R.baseline.mean(), beats_baseline=float((a > R.baseline).mean()),
        sd_binom=sd_binom, deff=deff, sd_binom_deff=sd_binom_deff,
        sd_ratio=a.std(ddof=1)/sd_binom_deff,
        spearman_val_test=rho, spearman_p=pv,
        frac_groups_extreme=float(((ga<0.05)|(ga>0.95)).mean())))
    pooled[lv] = R
S = pd.DataFrame(rows)
S.to_csv(PA/"summary.csv", index=False)
pd.set_option("display.width",200)
print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ── required number of families ──────────────────────────────────────────────
z = 1.959963985
p_star = float(S.loc[S.level=="family","mean"])
deff_f = float(S.loc[S.level=="family","deff"])
req = []
for E in [0.30,0.25,0.20,0.15,0.125,0.10,0.075,0.05,0.025]:
    n_ind = p_star*(1-p_star)*(z/E)**2
    req.append(dict(half_width=E, n_families_independent=n_ind,
                    n_families_with_deff=n_ind*deff_f))
RQ = pd.DataFrame(req); RQ.to_csv(PA/"required_n.csv", index=False)
print("\n== required families at p=%.3f (design effect %.2f) ==" % (p_star, deff_f))
print(RQ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ── model selection on validation is worthless at family level ───────────────
Rf = pooled["family"]
k = 10
rng = np.random.default_rng(0)
sel, best, rnd = [], [], []
for _ in range(2000):
    idx = rng.choice(len(Rf), k, replace=False)
    sub = Rf.iloc[idx]
    sel.append(sub.acc_test.iloc[int(np.argmax(sub.acc_val.values))])
    best.append(sub.acc_test.max()); rnd.append(sub.acc_test.iloc[0])
print(f"\n== picking the best of {k} splits by validation accuracy (family level) ==")
print(f"  select-on-val test acc : {np.mean(sel):.4f}")
print(f"  pick-at-random test acc: {np.mean(rnd):.4f}")
print(f"  oracle (best possible) : {np.mean(best):.4f}")
json.dump(dict(select_on_val=float(np.mean(sel)), random=float(np.mean(rnd)),
               oracle=float(np.mean(best)), k=k), open(PA/"selection.json","w"), indent=1)
