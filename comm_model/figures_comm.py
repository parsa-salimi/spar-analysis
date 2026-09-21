"""Figures for the communication-volume model."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import commvol as C

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID, SURF, RED = "#0b0b0b", "#52514e", "#8a8985", "#e3e2de", "#fcfcfb", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": GRID, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": .6, "legend.frameon": False, "figure.dpi": 150})
def clean(ax, axis="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis=axis)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)

df = pd.read_csv("grid.csv.gz", low_memory=False)
res = json.load(open("separability.json"))
gb = lambda x: x / 1e9

# ═════ F1 — the gate: does volume separate the classes? ═════
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3),
                              gridspec_kw={"width_ratios": [1.55, 1]})
# configurations with literally zero interconnect traffic are drawn separately:
# they are a BLIND SPOT, not a low value, and log-clamping them lies about that
nz = df[df.total_bps > 0]
n_zero = df.groupby("cls").apply(lambda g: int((g.total_bps <= 0).sum()))
bins = np.linspace(-1.2, 3.2, 70)
for cls, col, lab in [("training", BLUE, "training"), ("inference", ORANGE, "inference")]:
    v = np.log10(gb(nz[nz.cls == cls].total_bps.values))
    ax.hist(v, bins=bins, density=True, color=col, alpha=.55, label=lab, lw=0)
inf = gb(nz[nz.cls == "inference"].total_bps)
lo, hi = np.log10(max(inf.quantile(.05), 1e-2)), np.log10(inf.quantile(.95))
ax.axvspan(lo, hi, color=ORANGE, alpha=.07, lw=0, zorder=0)
ax.text((lo + hi) / 2, ax.get_ylim()[1] * .95,
        f"{res['train_inside_inference_90pct_band']:.0%} of training\nconfigurations sit inside\n"
        "inference's 5–95% band", ha="center", va="top", fontsize=8.5,
        color=INK, fontweight="bold")
ax.set_xticks([-1, 0, 1, 2, 3])
ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
ax.text(-1.15, ax.get_ylim()[1]*.52,
        f"plus {int(n_zero['inference']):,} serving and\n{int(n_zero['training'])} training "
        "configurations\nwith ZERO interconnect traffic —\na monitor cannot see either",
        fontsize=7.5, color=MUTED, va="top", ha="left", style="italic")
ax.set_xlabel("per-GPU interconnect throughput, GB/s  (log scale)")
ax.set_ylabel("density of configurations")
ax.set_title("Byte volume does not separate training from inference")
ax.legend(loc="upper left", fontsize=8.5); clean(ax)

modes = [("colocated", "vs co-located\nserving"),
         ("disaggregated", "vs disaggregated\nprefill/decode")]
vals = [res[f"volume_vs_{m}"] for m, _ in modes] + [res["total volume"]]
modes = modes + [("all", "vs all serving\nmodes pooled")]
cols = [RED if v < .75 else RAMP[3] for v in vals]
y = np.arange(len(modes))[::-1]
ax2.barh(y, vals, color=cols, height=.5, lw=0)
ax2.axvline(.5, color=INK2, lw=1.2, ls=(0, (3, 2)))
ax2.text(.5, y[0] + .45, "chance", fontsize=8, color=INK2, ha="center", va="bottom")
for yi, v in zip(y, vals):
    ax2.text(v + .015, yi, f"{v:.2f}", va="center", fontsize=9, fontweight="bold",
             color=RED if v < .75 else RAMP[4])
ax2.set_yticks(y); ax2.set_yticklabels([l for _, l in modes], fontsize=8)
ax2.set_xlim(0, 1.12); ax2.set_ylim(y[-1]-.55, y[0]+1.0)
ax2.set_xticks([0, .25, .5, .75, 1])
ax2.set_xlabel("AUC of a volume threshold")
ax2.set_title("A volume threshold is at chance"); clean(ax2, "x")
fig.text(.008, .015, f"{len(df):,} analytically-derived configurations across 9 models, 4 GPU types "
         "and realistic parallelism, batch, sequence and serving settings.\nThis is a feasibility bound "
         "under a stated configuration prior, NOT a measured detection rate.", fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .055, 1, 1])
for e in ("png", "pdf"): fig.savefig(OUT / f"c1_volume_overlap.{e}", bbox_inches="tight")
plt.close(fig)

# ═════ F2 — what a counter-based verifier can actually use ═════
tiers = [("volume only", "volume only (T1 minus symmetry)", RAMP[0], True),
         ("+ tx/rx symmetry", "T1  counters + tx/rx symmetry", RAMP[2], True),
         ("+ temporal structure", "T2  + temporal structure", RAMP[4], True),
         ("+ message size & count", "T3  + message size & count (NOT counter-observable)",
          MUTED, False)]
fig, ax = plt.subplots(figsize=(7.6, 3.9))
y = np.arange(len(tiers))[::-1]
for yi, (lab, key, col, obs) in zip(y, tiers):
    v = res[f"heldout_model::{key}"]
    ax.barh(yi, v, color=col, height=.5, lw=0, hatch=None if obs else "///",
            edgecolor=SURF)
    ax.text(v + .008, yi, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold",
            color=col if obs else INK2)
ax.axvline(.5, color=INK2, lw=1.2, ls=(0, (3, 2)))
ax.set_yticks(y); ax.set_yticklabels([t[0] for t in tiers], fontsize=9)
ax.set_xlim(0, 1.1); ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xlabel("held-out-model AUC  (train on 8 model families, score the 9th)")
ax.set_title("Symmetry is the feature that earns its place — and a counter can see it")
ax.set_ylim(y[-1] - 1.15, y[0] + .5)
ax.text(.02, y[-1] - .72, "hatched = not observable from cumulative counters; it needs an\n"
        "in-process hook, i.e. the adversary reporting on itself",
        ha="left", fontsize=7.5, color=MUTED)
clean(ax, "x")
fig.text(.008, .015, "tx and rx are separate hardware registers, so symmetry costs nothing extra to "
         "collect. Individual message sizes are not visible from cumulative counters at any sample rate.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .07, 1, 1])
for e in ("png", "pdf"): fig.savefig(OUT / f"c2_feature_tiers.{e}", bbox_inches="tight")
plt.close(fig)

# ═════ F3 — what each evasion costs ═════
ev = pd.read_csv("evasion_cost.csv")
style = {"DiLoCo": (BLUE, "o"), "gradient accumulation": (ORANGE, "s"),
         "traffic shaping": (AQUA, "^"), "run segmentation": (VIOLET, "D")}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                              gridspec_kw={"width_ratios": [1.35, 1]})
ax.axhspan(0.99, 1.08, color=RED, alpha=.07, lw=0)
ax.text(3.5e-4, .06, "shaded band = no wall-clock cost to the adversary", fontsize=8.5,
        color=RED, fontweight="bold", va="bottom", ha="left")
for name, g in ev.groupby("evasion"):
    col, mk = style[name]
    g = g.sort_values("inter_ratio")
    x = np.maximum(g.inter_ratio, 5e-4)
    ax.plot(x, g.throughput_retained, "-", color=col, lw=2, marker=mk, ms=6,
            markeredgecolor=SURF, markeredgewidth=1.2)
    i = int(np.argmin(g.throughput_retained.values))
    ax.annotate(name, (x.values[i], g.throughput_retained.values[i]),
                textcoords="offset points", xytext=(9, -13 if name=="traffic shaping" else 7),
                fontsize=8.5, color=col, fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(3e-4, 2.0); ax.set_ylim(0, 1.1)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("inter-node traffic remaining, fraction of baseline  →  quieter")
ax.set_ylabel("training throughput retained")
ax.set_title("What each evasion costs"); clean(ax)

# the decision-relevant number: how quiet can each get for free?
# reduction FACTOR, so a longer bar means a quieter run -- better for the adversary
free = (ev[ev.throughput_retained >= .99].groupby("evasion").inter_ratio.min()
          .reindex(list(style)).fillna(1.0))
red = 1.0 / np.maximum(free.values, 1e-6)
yy = np.arange(len(free))[::-1]
ax2.barh(yy, red, color=[style[k][0] for k in free.index], height=.5, lw=0, log=True)
for yi, k, v in zip(yy, free.index, red):
    ax2.text(v * 1.35, yi, "no reduction" if v < 1.05 else f"{v:,.0f}× quieter",
             va="center", fontsize=9, fontweight="bold",
             color=MUTED if v < 1.05 else style[k][0])
ax2.set_yticks(yy); ax2.set_yticklabels(free.index, fontsize=8.5)
ax2.set_xscale("log"); ax2.set_xlim(.7, 3e4)
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}×"))
ax2.set_xlabel("inter-node traffic reduction available at ≥99% throughput")
ax2.set_title("…and how much is free"); clean(ax2, "x")
fig.text(.008, .015, "Baseline: Llama-3-70B, FSDP, dp64 × tp8 on H100. Wall-clock cost only. The "
         "CONVERGENCE cost of DiLoCo and of gradient accumulation is not\ncomputable here and is "
         "deliberately left unmodelled — it has to be measured before either can be called expensive.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=[0, .065, 1, 1])
for e in ("png", "pdf"): fig.savefig(OUT / f"c3_evasion_cost.{e}", bbox_inches="tight")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("c*.png")))
