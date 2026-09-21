"""Figures for the measurement-artifact study."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent / "artifact_results"
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF, RED = "#0b0b0b", "#52514e", "#8a8985", "#e3e2de", "#fcfcfb", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": GRID, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": .6, "legend.frameon": False, "figure.dpi": 150})
def clean(ax, axis="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis=axis)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)

S = pd.read_csv(OUT / "conditions.csv")
P = pd.read_csv(OUT / "per_family.csv")
ctl = json.load(open(OUT / "controls.json"))
N_FAM = int(P[P.cond == "ceiling"].family.nunique())
RES = 1.0 / N_FAM                       # one family is worth this much AUC

# families correct per condition
base = P[P.cond == "ceiling"].set_index("family")
corr = {}
for c, g in P.groupby("cond"):
    g = g.set_index("family")
    ok = ((g.mean_score > .5) == (g.cls == "training"))
    corr[c] = int(ok.sum())
S["n_correct"] = S.cond.map(corr)

hz = S[S.knob == "sample_hz"].copy()
ceil_auc = float(S[S.cond == "ceiling"].auc.iloc[0])
ceil_ok = corr["ceiling"]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))

# ── (a) AUC vs sample rate ─────────────────────────────────────────────────
ax = axes[0]
ax.axhspan(ceil_auc - RES, ceil_auc + RES, color=MUTED, alpha=.13, lw=0)
ax.text(.17, ceil_auc - RES - .012, "within one family of the ceiling\n= not resolvable here",
        fontsize=7.2, color=MUTED, va="top")
ax.axhline(ceil_auc, color=INK2, lw=1.2, ls=(0, (3, 2)))
ax.text(52, ceil_auc + .012, "ceiling", fontsize=7.5, color=INK2, ha="right")
for w, col, mk, lab in ((64, BLUE, "o", "64-bit counters"), (32, ORANGE, "s", "32-bit word counters")):
    d = hz[hz.width == w].sort_values("x")
    ax.plot(d.x, d.auc, "-", color=col, lw=2, marker=mk, ms=5.5,
            markeredgecolor=SURF, markeredgewidth=1.1, label=lab)
    ax.fill_between(d.x, d.ci_lo, d.ci_hi, color=col, alpha=.12, lw=0)
ax.axhline(.5, color=RED, lw=1, ls=(0, (2, 2)))
ax.text(.17, .515, "chance", fontsize=7.5, color=RED)
ax.set_xscale("log"); ax.set_xlim(.15, 60); ax.set_ylim(.2, 1.03)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("sample rate, Hz"); ax.set_ylabel("held-out-family AUC")
ax.set_title("Detection vs how often we look")
ax.legend(loc="lower right", fontsize=8); clean(ax)

# ── (b) families correct -- the robust view ────────────────────────────────
ax = axes[1]
for w, col, mk, lab in ((64, BLUE, "o", "64-bit"), (32, ORANGE, "s", "32-bit words")):
    d = hz[hz.width == w].sort_values("x")
    ax.plot(d.x, d.n_correct, "-", color=col, lw=2, marker=mk, ms=5.5,
            markeredgecolor=SURF, markeredgewidth=1.1, label=lab)
ax.axhline(ceil_ok, color=INK2, lw=1.2, ls=(0, (3, 2)))
ax.text(52, ceil_ok + .12, f"ceiling: {ceil_ok}/{N_FAM}", fontsize=7.5, color=INK2, ha="right")
ax.axhline(N_FAM / 2, color=RED, lw=1, ls=(0, (2, 2)))
ax.text(.17, N_FAM / 2 + .12, "chance", fontsize=7.5, color=RED)
ax.set_xscale("log"); ax.set_xlim(.15, 60); ax.set_ylim(4.5, N_FAM + .6)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_yticks(range(5, N_FAM + 1))
ax.set_xlabel("sample rate, Hz"); ax.set_ylabel(f"workload families classified correctly (of {N_FAM})")
ax.set_title("…counted in whole workloads")
ax.legend(loc="lower right", fontsize=8); clean(ax)

# ── (c) clock skew: the feature dies, detection does not ───────────────────
ax = axes[2]
sk = S[S.knob == "clock_skew_ms"].sort_values("x").copy()
xc = []
for c in sk.cond:
    d = pd.read_parquet(OUT / "features" / f"{c}.parquet")
    xc.append(float(d[d.cls == "training"].xnode_corr.mean()))
sk["xnode"] = xc
x = np.maximum(sk.x.values, 0.3)        # 0 ms drawn at the left edge of the log axis
ax.plot(x, sk.xnode, "-", color=AQUA, lw=2.2, marker="^", ms=6,
        markeredgecolor=SURF, markeredgewidth=1.1,
        label="cross-node synchrony feature")
ax.plot(x, sk.auc, "-", color=BLUE, lw=2.2, marker="o", ms=6,
        markeredgecolor=SURF, markeredgewidth=1.1, label="detection (AUC)")
ax.fill_between(x, sk.ci_lo, sk.ci_hi, color=BLUE, alpha=.12, lw=0)
ax.axhline(.5, color=RED, lw=1, ls=(0, (2, 2)))
ax.set_xscale("log"); ax.set_ylim(-.05, 1.03)
ax.set_xticks([0.3, 1, 10, 100, 1000])
ax.set_xticklabels(["0", "1", "10", "100", "1000"])
ax.set_xlabel("clock skew between nodes, ms")
ax.set_ylabel("value (both on 0–1)")
ax.set_title("Clock skew kills the feature, not the detector")
ax.legend(loc="lower left", fontsize=8); clean(ax)

fig.text(.007, .015,
         f"{N_FAM} workload families x 3 seeds, {int(ctl['duration_s'])} s traces, "
         f"{int(ctl['window_s'])} s windows. Leave-one-family-out, scores pooled; bands are 95% CIs "
         f"bootstrapped over FAMILIES.\nControls: shuffled labels {ctl['floor_shuffled_auc']:.3f} "
         f"(must be 0.5); volume features alone {ctl['hardness_volume_only_auc']:.3f} "
         f"(the population is genuinely hard). One family is worth {RES:.3f} of AUC — "
         f"that is the study's resolution.", fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .085, 1, 1])
for e in ("png", "pdf"):
    fig.savefig(FIG / f"a1_measurement_artifacts.{e}", bbox_inches="tight")
plt.close(fig)
S.to_csv(OUT / "conditions_annotated.csv", index=False)
print("wrote", sorted(p.name for p in FIG.glob("a1*")))
print(f"resolution = 1 family = {RES:.3f} AUC; ceiling {ceil_auc:.3f} ({ceil_ok}/{N_FAM})")
