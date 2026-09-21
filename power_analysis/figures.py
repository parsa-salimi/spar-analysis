"""Figures + final numbers for the power analysis."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import config  # noqa: F401
from config import RESULTS as PA, HERE as OUT
(OUT/"figures").mkdir(parents=True, exist_ok=True)

# ── palette: grouping strictness is ORDINAL -> one-hue blue ramp, light->dark
RAMP   = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK    = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8985"
GRID   = "#e3e2de"; SURF = "#fcfcfb"; RED = "#e34948"; ORANGE = "#eb6834"
LEVELS = ["run", "job", "fine", "family", "coarse"]
NICE   = {"run": "run\n(one GPU stream)", "job": "job\n(one execution)",
          "fine": "workload\n(reps merged)", "family": "family\n(knobs collapsed)",
          "coarse": "mechanism\n(class only)"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.labelsize": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150,
})
def clean(ax, xgrid=False):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True); ax.grid(axis="x" if xgrid else "y")
    if not xgrid: ax.grid(axis="x", visible=False)
    else: ax.grid(axis="y", visible=False)

S = pd.read_csv(PA/"summary.csv").set_index("level").loc[LEVELS].reset_index()
R = {lv: pd.read_csv(PA/f"reps_{lv}.csv") for lv in LEVELS}

# ── empirical variance inflation phi (vs independent Bernoulli over groups) ──
for i, r in S.iterrows():
    S.loc[i, "phi"] = (r["sd"]**2) * r["n_grp_test"] / (r["mean"]*(1-r["mean"]))
    S.loc[i, "n_eff_test"] = r["mean"]*(1-r["mean"]) / (r["sd"]**2)

# ── LOFO ────────────────────────────────────────────────────────────────────
L = pd.read_csv(PA/"lofo_family.csv").drop_duplicates("group")
Lc = pd.read_csv(PA/"lofo_coarse.csv").drop_duplicates("group")
lofo = {}
for name, D in [("family", L), ("coarse", Lc)]:
    N = len(D); m = D.acc.mean(); se = D.acc.std(ddof=1)/np.sqrt(N)
    lofo[name] = dict(N=N, family_mean=float(m), se=float(se),
                      ci=float(1.96*se), window_weighted=float(np.average(D.acc, weights=D.n_win)),
                      frac_extreme=float(((D.acc < .05) | (D.acc > .95)).mean()))
print(json.dumps(lofo, indent=1))

# ═════════ F1 — accuracy vs grouping level ═════════
fig, ax = plt.subplots(figsize=(7.4, 4.3))
rng = np.random.default_rng(1)
for i, lv in enumerate(LEVELS):
    a = R[lv].acc_test.values
    ax.scatter(np.full(len(a), i) + rng.normal(0, .07, len(a)), a, s=9,
               color=RAMP[i], alpha=.35, linewidths=0, zorder=2)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    ax.plot([i-.28, i+.28], [med, med], color=RAMP[i], lw=2.4, zorder=4,
            solid_capstyle="round")
    ax.add_patch(plt.Rectangle((i-.28, q1), .56, q3-q1, facecolor="none",
                               edgecolor=RAMP[i], lw=1.4, zorder=3))
    b = S.loc[S.level == lv, "baseline"].iloc[0]
    ax.plot([i-.34, i+.34], [b, b], color=INK2, lw=1.4, ls=(0, (3, 2)), zorder=5)
    ax.text(i, 1.055, f"{a.mean():.2f}", ha="center", color=RAMP[i],
            fontweight="bold", fontsize=10)
    ax.text(i, 1.005, f"n={S.loc[S.level==lv,'n_groups'].iloc[0]}", ha="center",
            color=MUTED, fontsize=7.5)
ax.set_xticks(range(5)); ax.set_xticklabels([NICE[l] for l in LEVELS], fontsize=8)
ax.set_ylim(-.02, 1.12); ax.set_yticks(np.arange(0, 1.01, .2))
ax.set_ylabel("held-out accuracy (window level)")
ax.set_xlabel("what is held out  →  stricter", labelpad=8)
ax.set_title("The headline accuracy is a property of the split, not the detector")
ax.plot([], [], color=INK2, lw=1.4, ls=(0, (3, 2)), label="always-predict-training baseline")
ax.legend(loc="lower left", fontsize=8)
clean(ax)
fig.text(.012, .015, "Binary training-vs-rest, RF-100, datacenter corpus "
         "(A100-80GB / B200 / H200; 1,000 runs, 179 workloads). Each dot is one "
         "random group-level split; box = IQR, bar = median.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .045, 1, 1])
for e in ("png", "pdf"): fig.savefig(OUT/f"figures/f1_grouping_ladder.{e}", bbox_inches="tight")
plt.close(fig)

# ═════════ F2 — validation cannot rank ═════════
sel = json.load(open(PA/"selection.json"))
fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5))
for ax, lv, col in zip(axes, ["fine", "family", "coarse"], [RAMP[2], RAMP[3], RAMP[4]]):
    d = R[lv]; rho, pv = st.spearmanr(d.acc_val, d.acc_test)
    ax.scatter(d.acc_val, d.acc_test, s=14, color=col, alpha=.5, linewidths=0)
    lo, hi = .0, 1.0
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=.9, ls=(0, (3, 2)), zorder=1)
    b = np.polyfit(d.acc_val, d.acc_test, 1)
    xs = np.linspace(d.acc_val.min(), d.acc_val.max(), 2)
    ax.plot(xs, np.polyval(b, xs), color=RED if rho < 0 else col, lw=2)
    ax.set_title(f"{lv}   ρ = {rho:+.3f}  (p = {pv:.2f})", fontsize=9.5, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("validation accuracy")
    clean(ax)
axes[0].set_ylabel("test accuracy")
fig.suptitle("Validation accuracy carries no information about test accuracy",
             x=.012, ha="left", fontsize=10.5, fontweight="bold")
fig.text(.012, .015, f"Picking the best of 10 splits by validation accuracy (family level) "
         f"gives test {sel['select_on_val']:.3f} — worse than picking at random "
         f"({sel['random']:.3f}); the oracle would give {sel['oracle']:.3f}. "
         f"Dashed line is y = x.", fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .06, 1, .93])
for e in ("png", "pdf"): fig.savefig(OUT/f"figures/f2_val_cannot_rank.{e}", bbox_inches="tight")
plt.close(fig)

# ═════════ F3 — required families ═════════
p = lofo["family"]["family_mean"]
phi = float(S.loc[S.level == "family", "phi"].iloc[0])
z = 1.959963985
E = np.linspace(.03, .32, 300)
n_ind = p*(1-p)*(z/E)**2
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(E, n_ind, color=RAMP[1], lw=2, label="independent families (lower bound)")
ax.plot(E, n_ind*phi, color=RAMP[4], lw=2.4,
        label=f"measured between-family heterogeneity (×{phi:.2f})")
ax.axhline(lofo["family"]["N"], color=ORANGE, lw=1.6, ls=(0, (3, 2)))
ax.text(.318, lofo["family"]["N"]*1.18, f"corpus today: {lofo['family']['N']} families",
        ha="left", color=ORANGE, fontsize=8, fontweight="bold")
for e_mark, off in ((.20, (8, -13)), (.10, (10, 6)), (.05, (10, 6))):
    n_mark = p*(1-p)*(z/e_mark)**2*phi
    ax.plot([e_mark], [n_mark], "o", ms=6, color=RAMP[4], zorder=5,
            markeredgecolor=SURF, markeredgewidth=1.4)
    ax.annotate(f"±{e_mark:.2f} → {n_mark:,.0f}", (e_mark, n_mark),
                textcoords="offset points", xytext=off, fontsize=8,
                color=RAMP[4], fontweight="bold")
ax.set_yscale("log"); ax.set_xlim(.32, .03)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlabel("95% CI half-width on the reported accuracy  →  more precise")
ax.set_ylabel("workload families required (log scale)")
ax.set_title("What precision costs, measured in workload families")
ax.legend(loc="upper left", fontsize=8)
clean(ax)
fig.text(.012, .015, f"Leave-one-family-out, p = {p:.3f}. Families, not windows, "
         f"are the unit of replication: window count does not enter.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .045, 1, 1])
for e in ("png", "pdf"): fig.savefig(OUT/f"figures/f3_required_families.{e}", bbox_inches="tight")
plt.close(fig)

# ═════════ F4 — LOFO per-family ═════════
D = L.sort_values("acc").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
cols = [RED if a < .5 else RAMP[2] for a in D.acc]
ax.bar(range(len(D)), D.acc, color=cols, width=.78, linewidth=0)
m = lofo["family"]["family_mean"]; ci = lofo["family"]["ci"]
ax.axhline(m, color=INK, lw=1.6)
ax.axhspan(m-ci, m+ci, color=INK, alpha=.07, lw=0)
ax.text(len(D)-.5, m+.03, f"mean {m:.3f} ± {ci:.3f}  (95% CI)", ha="right",
        color=INK, fontsize=8.5, fontweight="bold")
ax.set_xlim(-0.7, len(D)-0.3); ax.set_ylim(0, 1.02)
ax.set_xlabel(f"workload family, sorted  (n = {len(D)})")
ax.set_ylabel("accuracy when that family is held out")
ax.set_title("A family is decided as a unit — so it counts as one observation, not thousands")
ax.set_xticks([])
n_zero = int((D.acc < .05).sum()); n_one = int((D.acc > .95).sum())
fig.text(.012, .015, f"{n_zero} of {len(D)} families are classified essentially 0% "
         f"correct and {n_one} essentially 100% — {(n_zero+n_one)/len(D):.0%} of families are "
         f"decided all-or-nothing. Red = worse than a coin flip on that family.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .045, 1, 1])
for e in ("png", "pdf"): fig.savefig(OUT/f"figures/f4_lofo_families.{e}", bbox_inches="tight")
plt.close(fig)

S.to_csv(OUT/"summary_levels.csv", index=False)
json.dump(dict(lofo=lofo, phi=phi, selection=sel), open(OUT/"headline_numbers.json", "w"), indent=1)
print(S[["level","n_groups","mean","sd","baseline","beats_baseline","phi","n_eff_test",
         "spearman_val_test"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("figures ->", OUT/"figures")
