"""Read the frozen forests. Do not refit them.

Walks every split in frozen/stage2_gpu.joblib and frozen/fallback_gpu.joblib,
weights each split by the impurity it removes, and writes a short PDF plus the
per-feature table.

    python -m classifier_analysis.frozen_interp

Needs the environment the freeze was built with (scikit-learn 1.8) so the
joblib files load as the same trees. matplotlib and reportlab draw the note.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import common as C
from .detector_frozen import load

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = Path("/tmp/frozen_interp_figs")

INK = "#1F2933"
MUTED = "#52606D"
STAGE = "#1B4F72"
FALL = "#B85C38"
GREEN = "#2F6F4E"
ROSE = "#9B3A4A"
GRAY = "#7B8794"
PALE = "#F4F6F8"
LINE = "#E4E7EB"

NAMES = {
    "mem_utilization_pct_std": "Memory utilization, spread",
    "mem_utilization_pct_range": "Memory utilization, range",
    "mem_utilization_pct_iqr": "Memory utilization, interquartile range",
    "mem_utilization_pct_cv": "Memory utilization, coefficient of variation",
    "time_to_mem_plateau_samples": "Samples until memory plateaus",
    "mem_used_mb_max": "Memory in use, maximum",
    "mem_used_mb_p95": "Memory in use, 95th percentile",
    "mem_used_mb_p75": "Memory in use, 75th percentile",
    "mem_used_mb_p50": "Memory in use, median",
    "mem_used_mb_mean": "Memory in use, mean",
    "power_draw_w_std": "Power, spread",
    "power_draw_w_range": "Power, range",
    "temperature_c_std": "Temperature, spread",
    "temperature_c_range": "Temperature, range",
    "first_30s_mem_delta_mb": "Startup memory increase",
    "first_30s_mem_ratio": "Startup memory, peak / start",
    "first_30s_mem_delta_frac": "Startup memory, fraction of capacity",
    "mem_util_minus_gpu_util": "Memory utilization minus GPU utilization",
    "mem_util_spike": "Memory-utilization spike",
    "pcie_tx_mbps_range": "PCIe transmit, range",
    "gpu_utilization_pct_mean": "GPU utilization, mean",
}


def _p_train(val, node: int) -> float:
    v = np.asarray(val[node]).reshape(-1)
    s = float(v.sum())
    return float(v[1] / s) if s else float("nan")


def analyze(model) -> tuple[pd.DataFrame, dict]:
    n = model.n_features_in_
    cols = list(pd.read_parquet(HERE / "frozen" / "gpu_cols.parquet").col)
    if len(cols) != n:
        raise RuntimeError(f"feature list is {len(cols)} long, forest has {n}")
    dec = np.zeros(n)
    dec_d0 = np.zeros(n)
    nsplit = np.zeros(n, dtype=int)
    nroot = np.zeros(n, dtype=int)
    dir_w = np.zeros(n)
    dir_m = np.zeros(n)
    thr_w: list[list[tuple[float, float]]] = [[] for _ in range(n)]
    depths, leaves = [], []
    for est in model.estimators_:
        t = est.tree_
        feat, thr = t.feature, t.threshold
        left, right = t.children_left, t.children_right
        imp, wn, val = t.impurity, t.weighted_n_node_samples, t.value
        depth = np.full(t.node_count, -1)
        depth[0] = 0
        stack = [0]
        while stack:
            i = stack.pop()
            for c in (int(left[i]), int(right[i])):
                if c >= 0:
                    depth[c] = depth[i] + 1
                    stack.append(c)
        depths.append(int(depth.max()))
        leaves.append(int((feat < 0).sum()))
        for i in range(t.node_count):
            f = int(feat[i])
            if f < 0:
                continue
            L, R = int(left[i]), int(right[i])
            d = float(wn[i] * imp[i] - wn[L] * imp[L] - wn[R] * imp[R])
            if d < 0:
                d = 0.0
            dec[f] += d
            nsplit[f] += 1
            if int(depth[i]) == 0:
                dec_d0[f] += d
                nroot[f] += 1
            delta = _p_train(val, R) - _p_train(val, L)
            dir_w[f] += d
            dir_m[f] += d * delta
            thr_w[f].append((float(thr[i]), d))
    total = float(dec.sum())
    rows = []
    for j, c in enumerate(cols):
        if dec[j] == 0 or dir_w[j] == 0:
            wmed, direction = float("nan"), float("nan")
        else:
            ts, ws = zip(*thr_w[j])
            order = np.argsort(ts)
            ts, ws = np.asarray(ts)[order], np.asarray(ws)[order]
            cw = np.cumsum(ws)
            wmed = float(ts[min(np.searchsorted(cw, 0.5 * cw[-1]), len(ts) - 1)])
            direction = float(dir_m[j] / dir_w[j])
        rows.append(dict(
            feature=c, group=C.group_of(c),
            impurity=dec[j] / total,
            root_share=dec_d0[j] / float(dec_d0.sum()),
            n_splits=int(nsplit[j]), n_roots=int(nroot[j]),
            vote_gap=direction, threshold=wmed,
        ))
    df = pd.DataFrame(rows)
    gap = float(np.max(np.abs(model.feature_importances_ - df.impurity.values)))
    meta = dict(
        mean_depth=float(np.mean(depths)), max_depth=int(np.max(depths)),
        mean_leaves=float(np.mean(leaves)),
        features_used=int((df.n_splits > 0).sum()),
        n_features=n, max_features=int(model.estimators_[0].max_features_),
        importance_gap=gap,
    )
    return df, meta


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D9DEE5")
    ax.spines["bottom"].set_color("#D9DEE5")
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_facecolor("white")


def fig_groups(stage, fall, path: Path):
    order = ["Memory", "Power & thermal", "PCIe", "Compute", "Video codec"]
    s = stage.groupby("group").impurity.sum().reindex(order).fillna(0)
    f = fall.groupby("group").impurity.sum().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(7.15, 3.15), dpi=160)
    y = np.arange(len(order))
    h = 0.36
    ax.barh(y + h / 2, s.values * 100, h, color=STAGE, label="Stage 2  (rule fired)")
    ax.barh(y - h / 2, f.values * 100, h, color=FALL, label="Fallback  (rule silent)")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Share of impurity removed  (%)")
    ax.set_xlim(0, 78)
    ax.invert_yaxis()
    for i, (a, b) in enumerate(zip(s.values, f.values)):
        ax.text(a * 100 + 1.0, i + h / 2, f"{a * 100:.0f}", va="center", ha="left", color=STAGE, fontsize=8)
        ax.text(b * 100 + 1.0, i - h / 2, f"{b * 100:.0f}", va="center", ha="left", color=FALL, fontsize=8)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_features(df: pd.DataFrame, title: str, color_pos: str, path: Path, n=8):
    top = df.sort_values("impurity", ascending=False).head(n).iloc[::-1]
    labels = [NAMES.get(f, f) for f in top.feature]
    colors = [GREEN if g > 0.2 else ROSE if g < -0.2 else GRAY for g in top.vote_gap]
    fig, ax = plt.subplots(figsize=(7.15, 3.7), dpi=160)
    y = np.arange(len(top))
    ax.barh(y, top.impurity.values * 100, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of impurity removed  (%)")
    ax.set_xlim(0, max(top.impurity.values) * 100 * 1.38)
    for i, (imp, gap, roots) in enumerate(zip(top.impurity, top.vote_gap, top.n_roots)):
        side = "high side votes training" if gap > 0 else "high side votes not training"
        ax.text(imp * 100 + 0.15, i, f"{imp * 100:.1f}    gap {gap:+.2f}    {int(roots)} roots",
                va="center", ha="left", color=INK, fontsize=7.5)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    del title, color_pos


def fig_footprint(stage_gb: float, fall_gb: float, path: Path):
    fig, ax = plt.subplots(figsize=(7.15, 2.05), dpi=160)
    ax.set_xlim(0, 84)
    ax.set_ylim(0, 1)
    ax.axhline(0.22, color="#D9DEE5", lw=6, solid_capstyle="round", zorder=1)
    ax.plot([stage_gb], [0.22], "o", ms=10, color=STAGE, zorder=3)
    ax.plot([fall_gb], [0.22], "o", ms=10, color=FALL, zorder=3)
    ax.text(stage_gb + 2.2, 0.78, f"Stage 2: above {stage_gb:.0f} GB, votes training",
            color=STAGE, fontsize=8.5, ha="left", va="center")
    ax.annotate("", xy=(stage_gb, 0.28), xytext=(stage_gb, 0.68),
                arrowprops=dict(arrowstyle="-|>", color=STAGE, lw=0.8))
    ax.text(fall_gb - 2.2, 0.52, f"Fallback: above {fall_gb:.0f} GB, votes not training",
            color=FALL, fontsize=8.5, ha="right", va="center")
    ax.annotate("", xy=(fall_gb, 0.28), xytext=(fall_gb, 0.46),
                arrowprops=dict(arrowstyle="-|>", color=FALL, lw=0.8))
    ax.set_xlabel("GPU memory in use, maximum inside the window  (GB)")
    ax.set_yticks([])
    _style_ax(ax)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_gate(path: Path):
    fig, ax = plt.subplots(figsize=(7.15, 2.55), dpi=160)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    def box(x, y, w, h, face, edge, text):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)

    box(0.15, 1.85, 2.35, 1.45, "#F7F8FA", INK, "A 30 s window")
    box(3.35, 1.85, 2.9, 1.45, "#E7F0F7", STAGE, "Network rule\n>= 1 MiB/s and\nsymmetry >= 0.8")
    box(8.15, 3.15, 3.8, 1.6, "#E7F0F7", STAGE, "Stage 2 forest\nmemory and power\nare moving")
    box(8.15, 0.4, 3.8, 1.6, "#F8EFEA", FALL, "Fallback forest\nmemory jumped\nat startup")
    ax.annotate("", xy=(3.35, 2.57), xytext=(2.5, 2.57),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.annotate("", xy=(8.15, 3.95), xytext=(6.25, 3.05),
                arrowprops=dict(arrowstyle="-|>", color=STAGE, lw=1.1))
    ax.annotate("", xy=(8.15, 1.2), xytext=(6.25, 2.1),
                arrowprops=dict(arrowstyle="-|>", color=FALL, lw=1.1))
    ax.text(6.45, 4.55, "fires", color=STAGE, fontsize=8.5, ha="left")
    ax.text(6.45, 0.55, "silent", color=FALL, fontsize=8.5, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _styles():
    s = {}
    s["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=18, leading=22,
                              textColor=HexColor(INK), spaceAfter=4)
    s["deck"] = ParagraphStyle("deck", fontName="Helvetica", fontSize=10.5, leading=14.5,
                                textColor=HexColor(MUTED), spaceAfter=8)
    s["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                              textColor=HexColor(INK), spaceBefore=8, spaceAfter=4)
    s["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13,
                                textColor=HexColor(INK), spaceAfter=6)
    s["cap"] = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8, leading=10.5,
                               textColor=HexColor(MUTED), spaceBefore=2, spaceAfter=6)
    s["cell"] = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.6, leading=9.6,
                                textColor=HexColor(INK))
    s["cellh"] = ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=7.6, leading=9.6,
                                 textColor=HexColor(INK))
    s["foot"] = ParagraphStyle("foot", fontName="Helvetica", fontSize=8, leading=10.5,
                                textColor=HexColor(MUTED))
    return s


def _p(text, style):
    return Paragraph(text, style)


def _img(path: Path, width=7.15 * inch):
    im = Image(str(path), width=width, height=width * _aspect(path))
    im.hAlign = "LEFT"
    return im


def _aspect(path: Path) -> float:
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return h / w


def build_pdf(stage, fall, sm, fm, paths, pdf_path: Path):
    st = _styles()
    sg = stage.groupby("group").impurity.sum()
    fg = fall.groupby("group").impurity.sum()

    def row(feature, frame):
        return frame.loc[frame.feature == feature].iloc[0]

    s_std = row("mem_utilization_pct_std", stage)
    s_pow = row("power_draw_w_std", stage)
    s_mem = row("mem_used_mb_max", stage)
    s_plat = row("time_to_mem_plateau_samples", stage)
    f_d = row("first_30s_mem_delta_mb", fall)
    f_r = row("first_30s_mem_ratio", fall)
    f_f = row("first_30s_mem_delta_frac", fall)
    f_diff = row("mem_util_minus_gpu_util", fall)
    f_mem = row("mem_used_mb_max", fall)
    f_util = row("gpu_utilization_pct_mean", fall)
    startup = fall[fall.feature.isin([
        "first_30s_mem_delta_mb", "first_30s_mem_ratio", "first_30s_mem_delta_frac"])]
    spread = stage[stage.feature.isin([
        "mem_utilization_pct_std", "mem_utilization_pct_range", "mem_utilization_pct_iqr"])]

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        leftMargin=0.65 * inch, rightMargin=0.65 * inch,
        topMargin=0.62 * inch, bottomMargin=0.58 * inch,
        title="What the frozen forests split on",
        author="spar-analysis",
    )
    story = []
    story.append(_p("What the frozen forests split on", st["h1"]))
    story.append(_p(
        "A reading of the two random forests frozen on 12 October 2026. "
        "The trees were opened and counted. They were not retrained.",
        st["deck"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor(LINE), spaceAfter=8))

    story.append(_p("What we did", st["h2"]))
    story.append(_p(
        "The detector has a network rule and two forests on the same 174 GPU features. "
        "The rule fires when a node's mean send+receive rate is at least 1 MiB/s and the "
        "two directions are balanced (the smaller is at least 0.8 of the larger). "
        "Stage 2 was fit only on windows where that rule fires: 3,019 training windows and "
        "939 others, out of 3,958. Fallback was fit on every honest window: 10,059 windows, "
        "244 jobs, 42 families. A window is called training when the forest that is allowed "
        "to speak has probability at least 0.5. Adversarial jobs were not in the fit.",
        st["body"]))
    story.append(_img(paths["gate"]))
    story.append(_p(
        "Figure 1. Which forest is asked. Once the rule fires, fallback is not consulted.",
        st["cap"]))
    story.append(_p(
        f"Each forest is 200 trees, minimum 2 samples in a leaf, {sm['max_features']} features "
        f"drawn at each split, class weights balanced, seed 0. For every split we kept the "
        f"feature, the threshold, and which side receives the training vote. A split's weight "
        f"is the impurity it removes. That is the same weight as sklearn's feature importance: "
        f"the two agree to {max(sm['importance_gap'], fm['importance_gap']):.4f}. "
        f"Standard deviation, range, and interquartile range of one column are one measurement "
        f"written three times, so the shares are also added up by source.",
        st["body"]))
    story.append(_p(
        "This is not a permutation test. Shuffling a column on held-out jobs needs the training "
        "table, and that table is not what this note uses. The vote gap below is the tree's own "
        "class fraction on the high side of a cut, minus the fraction on the low side, averaged "
        "with the impurity weights. A gap near +1 means the high side votes training and the "
        "low side does not. Thresholds are impurity-weighted medians of many split points, not "
        "a single rule.",
        st["body"]))

    story.append(_p("The two forests are not the same size", st["h2"]))
    cards = [[
        _p(f"<font color='{STAGE}'><b>Stage 2</b></font>", st["body"]),
        _p(f"<font color='{FALL}'><b>Fallback</b></font>", st["body"]),
    ], [
        _p(f"Mean depth {sm['mean_depth']:.1f}, deepest {sm['max_depth']}. "
           f"About {sm['mean_leaves']:.0f} leaves per tree. "
           f"{sm['features_used']} of {sm['n_features']} features are split on at least once.",
           st["cell"]),
        _p(f"Mean depth {fm['mean_depth']:.1f}, deepest {fm['max_depth']}. "
           f"About {fm['mean_leaves']:.0f} leaves per tree. "
           f"{fm['features_used']} of {fm['n_features']} features are split on at least once.",
           st["cell"]),
    ]]
    card = Table(cards, colWidths=[3.55 * inch, 3.55 * inch])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#E7F0F7")),
        ("BACKGROUND", (1, 0), (1, -1), HexColor("#F8EFEA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (0, -1), 0.4, HexColor(STAGE)),
        ("BOX", (1, 0), (1, -1), 0.4, HexColor(FALL)),
    ]))
    story.append(card)
    story.append(Spacer(1, 8))
    story.append(_p(
        "Stage 2 is a short tree. Fallback is a real one. They also cut on different facts, "
        "which is the rest of this note.",
        st["body"]))

    story.append(CondPageBreak(4.2 * inch))
    story.append(_p("Where the impurity goes", st["h2"]))
    story.append(_img(paths["groups"]))
    story.append(_p(
        "Figure 2. Share of impurity removed, added up by the physical source of the column. "
        "Video-codec columns never vary in this corpus, so both forests ignore them.",
        st["cap"]))
    s_pow_share = float(sg.get("Power & thermal", 0)) * 100
    f_pow_share = float(fg.get("Power & thermal", 0)) * 100
    story.append(_p(
        f"Stage 2 puts {sg.get('Memory', 0) * 100:.0f}% on memory and "
        f"{s_pow_share:.0f}% on power and temperature. "
        f"PCIe is {sg.get('PCIe', 0) * 100:.0f}%. Compute, which is where GPU utilization "
        f"and SM clock live, is {sg.get('Compute', 0) * 100:.0f}%, and none of that is at the "
        f"root. GPU-utilization mean is rank "
        f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'gpu_utilization_pct_mean'].iloc[0])} "
        f"and removes essentially no impurity.",
        st["body"]))
    story.append(_p(
        f"Fallback is still {fg.get('Memory', 0) * 100:.0f}% memory, but the memory it uses "
        f"is the allocation at the start of the run. Compute rises to "
        f"{fg.get('Compute', 0) * 100:.0f}%, PCIe to {fg.get('PCIe', 0) * 100:.0f}%, and power "
        f"falls to {f_pow_share:.0f}%.",
        st["body"]))

    story.append(_p("Stage 2: a moving trace votes training", st["h2"]))
    story.append(_img(paths["stage"]))
    story.append(_p(
        "Figure 3. The eight heaviest features in stage 2. Green means the high side of the "
        "cut votes training. Several of these bars are one column: spread, range, and the "
        "interquartile range of memory utilization. "
        f"Those three are the root of {int(spread.n_roots.sum())} of the 200 trees. "
        f"Each split is shown only {sm['max_features']} of the {sm['n_features']} features, "
        "so a measurement that takes the root about as often as it is offered is one the "
        "tree actually prefers.",
        st["cap"]))
    story.append(KeepTogether([_p(
        f"The cuts are monotone. Memory-utilization spread above about "
        f"{s_std.threshold:.1f} percentage points (vote gap {s_std.vote_gap:+.2f}). "
        f"Power spread above about {s_pow.threshold:.0f} W (gap {s_pow.vote_gap:+.2f}). "
        f"Memory in use above about {s_mem.threshold / 1024:.0f} GB (gap {s_mem.vote_gap:+.2f}). "
        f"Time to the memory plateau longer than about {s_plat.threshold:.0f} samples "
        f"(gap {s_plat.vote_gap:+.2f}). The startup-allocation features that dominate fallback "
        f"rank between "
        f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'first_30s_mem_ratio'].iloc[0])} "
        f"and "
        f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'first_30s_mem_delta_mb'].iloc[0])} "
        f"here.",
        st["body"])]))
    story.append(_p(
        "The not-training windows in this fit are jobs with balanced traffic and a flat GPU "
        "trace. In the corpus those are HPC jobs, not inference. A serving window whose memory "
        "utilization moves is on the training side of every large cut. Stage 2 has no leaf "
        "whose job is to recognize inference.",
        st["body"]))

    story.append(CondPageBreak(4.4 * inch))
    story.append(_p("Fallback: a startup allocation votes training", st["h2"]))
    story.append(_img(paths["fall"]))
    story.append(_p(
        "Figure 4. The eight heaviest features in fallback. Green: high side votes training. "
        "Red: high side votes not training. Gray: the two sides barely disagree. "
        "The first three bars are the same event. "
        f"Together they are {startup.impurity.sum() * 100:.0f}% of the impurity and "
        f"{startup.root_share.sum() * 100:.0f}% of the root weight.",
        st["cap"]))
    story.append(_p(
        "In the feature code these columns read the first 30 samples of the run, as far as "
        "the current window is allowed to see. After those 30 samples, later windows of the "
        "same job all see the same jump. The comment on the function says it was written to "
        "catch optimizer state allocated at startup, on the order of 6 GB, which an inference "
        "process was not expected to do.",
        st["body"]))
    story.append(_p(
        f"Typical cuts: an increase above about {f_d.threshold / 1024:.1f} GB "
        f"(gap {f_d.vote_gap:+.2f}), a peak/start ratio above about {f_r.threshold:.0f} "
        f"(gap {f_r.vote_gap:+.2f}), or an increase above "
        f"{f_f.threshold:.2f} of capacity (gap {f_f.vote_gap:+.2f}).",
        st["body"]))
    story.append(_p(
        f"The next cuts point the other way. Memory utilization minus GPU utilization has "
        f"vote gap {f_diff.vote_gap:+.2f} and a typical cut near {f_diff.threshold:.0f} "
        f"percentage points: when the memory bus is busy relative to the SMs, the tree votes "
        f"not training. Absolute footprint flips relative to stage 2. Fallback's typical cut "
        f"on the maximum memory in use is {f_mem.threshold / 1024:.0f} GB, and the high side "
        f"votes not training (gap {f_mem.vote_gap:+.2f}). Stage 2's cut on the same column is "
        f"{s_mem.threshold / 1024:.0f} GB, high side training.",
        st["body"]))
    story.append(KeepTogether([
        _img(paths["foot"]),
        _p(
            "Figure 5. The same column, two forests, two directions. Stage 2 only ever saw "
            "balanced-traffic windows, where a footprint past about 10 GB separated training from "
            "flat HPC. Fallback also saw large serving jobs, and past about 68 GB it votes the "
            "other way. Both numbers are weighted medians of split points, not class medians.",
            st["cap"]),
    ]))
    story.append(_p(
        f"GPU utilization is not the feature either forest leads with. In fallback its mean "
        f"is rank {int(fall.impurity.rank(ascending=False)[fall.feature == 'gpu_utilization_pct_mean'].iloc[0])}, "
        f"the typical cut is {f_util.threshold:.1f}%, and the vote gap is only "
        f"{f_util.vote_gap:+.2f}. A GPU has to be pinned before this column counts.",
        st["body"]))

    story.append(_p("What that does to the gate", st["h2"]))
    story.append(_p(
        "On the 93 Nebius jobs already scored with this frozen detector, the pattern matches "
        "the cuts. Those jobs were not used to choose features or to move the 0.5 cutoff. "
        "Probabilities below are the mean over the job's windows.",
        st["body"]))

    header = ["Workload", "Rule", "Stage 2", "Fallback", "Call"]
    raw = [
        ["Training, IB off\n(iboff_flat)", "silent", "not asked", "0.65", "training"],
        ["Training, no gradient\ncollective (nosync_flat)", "silent", "not asked", "0.62 to 0.66", "training"],
        ["LoRA, one node, IB off\n(lora_1n8_iboff)", "silent", "not asked", "0.53 to 0.54", "training on 85% of windows"],
        ["Qwen serving plus a fake\nall-reduce (qwen_fake_ar)", "fires", "0.90 to 0.93", "0.30 to 0.37", "training"],
        ["OLMoE across nodes\n(olmoe_ep_mixed)", "fires", "0.89 to 0.90", "0.11 to 0.13", "training"],
        ["Split prefill/decode plus\na return message (disagg_echo)", "fires on most windows", "0.91 to 0.95", "0.35 to 0.47", "training"],
    ]
    data = [[_p(h, st["cellh"]) for h in header]]
    for r in raw:
        data.append([_p(c.replace("\n", "<br/>"), st["cell"]) for c in r])
    colw = [1.85 * inch, 1.15 * inch, 0.95 * inch, 1.15 * inch, 2.0 * inch]
    tbl = Table(data, colWidths=colw, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(PALE)),
        ("BACKGROUND", (0, 1), (-1, 3), HexColor("#F8EFEA")),
        ("BACKGROUND", (0, 4), (-1, 6), HexColor("#E7F0F7")),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)
    story.append(_p(
        "Table 1. Orange rows are decided by fallback, because the rule is silent. Blue rows "
        "are decided by stage 2, because the rule fired. Fallback's probability on the blue "
        "rows is below 0.5 and is ignored.",
        st["cap"]))
    story.append(_p(
        "The orange rows are training processes whose cross-node all-reduce is missing. "
        "Fallback sees the startup allocation and calls them training. That is the feature "
        "doing what it was built for. The blue rows are inference with balanced traffic on "
        "the wire. Stage 2 sees memory moving, the way it saw training, and calls them "
        "training. On every one of the 1,032 windows in this score where the rule fired, "
        "stage 2's probability was at least 0.62. It never vetoed.",
        st["body"]))
    story.append(_p(
        "Encoder utilization, decoder utilization, and the variation of the memory clock are "
        "never split on in either forest. Those columns are constant in the training corpus.",
        st["body"]))

    story.append(_p("What this note is not", st["h2"]))
    story.append(_p(
        "It does not say that memory spread causes a window to be training. It says that is "
        "the cut the saved trees use. Correlated copies of one signal share the impurity, "
        "which is why Figure 2 is the total to quote and Figures 3 and 4 are the detail. "
        "Nothing here refits a forest, adds the October campaigns, or retunes 0.5. "
        "The counting script is classifier_analysis/frozen_interp.py. "
        "Run it with scikit-learn 1.8 so the joblib files load as the same trees.",
        st["body"]))

    def _page(canvas, doc_):
        canvas.saveState()
        w, h = letter
        canvas.setFillColor(HexColor(STAGE))
        canvas.rect(0, h - 7, w, 7, fill=1, stroke=0)
        canvas.setFillColor(HexColor(MUTED))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.65 * inch, 0.34 * inch,
                           "spar-analysis   ·   frozen 12 October 2026   ·   trees read, not refit")
        canvas.drawRightString(w - 0.65 * inch, 0.34 * inch, str(doc_.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=_page, onLaterPages=_page)


def main():
    stage2, fallback, _cols = load()
    stage, sm = analyze(stage2)
    fall, fm = analyze(fallback)
    if sm["importance_gap"] > 1e-3 or fm["importance_gap"] > 1e-3:
        raise RuntimeError(f"impurity walk disagrees with sklearn: {sm['importance_gap']} {fm['importance_gap']}")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    both = pd.concat([
        stage.assign(forest="stage2"),
        fall.assign(forest="fallback"),
    ], ignore_index=True)
    csv_path = OUT / "frozen_interp_features.csv"
    both.to_csv(csv_path, index=False)
    paths = {
        "groups": FIG / "groups.png",
        "stage": FIG / "stage2.png",
        "fall": FIG / "fallback.png",
        "foot": FIG / "footprint.png",
        "gate": FIG / "gate.png",
    }
    fig_groups(stage, fall, paths["groups"])
    fig_features(stage, "stage2", STAGE, paths["stage"])
    fig_features(fall, "fallback", FALL, paths["fall"])
    fig_footprint(
        float(stage.loc[stage.feature == "mem_used_mb_max", "threshold"].iloc[0]) / 1024,
        float(fall.loc[fall.feature == "mem_used_mb_max", "threshold"].iloc[0]) / 1024,
        paths["foot"],
    )
    fig_gate(paths["gate"])
    pdf_path = OUT / "frozen_forest_interp.pdf"
    build_pdf(stage, fall, sm, fm, paths, pdf_path)
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")
    print("stage2", sm)
    print("fallback", fm)


if __name__ == "__main__":
    main()
