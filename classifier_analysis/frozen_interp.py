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
    depth_m = np.zeros(n)
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
            depth_m[f] += d * int(depth[i])
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
            mean_depth=(depth_m[j] / dec[j]) if dec[j] else float("nan"),
            root_frac=(dec_d0[j] / dec[j]) if dec[j] else float("nan"),
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


# One recorded window from each side of the gate, walked through one tree.
# Recomputed from the published parquet on 19 Oct 2026. Tree indexes are into
# the frozen estimators_ list. Thresholds are read back from the joblib when
# the figure is drawn, and checked against these.
STAGE_WALK = dict(
    tree=30, root="mem_utilization_pct_std",
    cell="Qwen serving plus a fake all-reduce",
    values={
        "mem_utilization_pct_std": 4.711552508385713,
        "mem_used_mb_kurt": 0.0,
        "mem_used_mb_max": 68990.0,
    },
    votes=187, proba=0.937, other_proba=0.318, mibps=81.4, sym=1.0,
)
FALL_WALK = dict(
    tree=0, root="first_30s_mem_ratio",
    cell="Training with InfiniBand off",
    values={
        "first_30s_mem_ratio": 10508.0,
        "pcie_rx_mbps_p95": 438.20544010404205,
        "mem_clock_mhz_p50": 2619.0,
        "temperature_c_mean": 67.97777777777779,
        "temperature_c_p75": 68.15833333333333,
    },
    votes=127, proba=0.626, mibps=0.0,
)

SHORT = {
    "mem_utilization_pct_std": "util spread",
    "mem_used_mb_mean": "memory, mean",
    "mem_used_mb_max": "memory in use",
    "mem_used_mb_kurt": "memory kurtosis",
    "sm_clock_mhz_p75": "SM clock",
    "temperature_c_p25": "temperature",
    "temperature_c_mean": "temperature",
    "temperature_c_p75": "temperature",
    "pcie_rx_mbps_p95": "PCIe receive",
    "pcie_tx_mbps_std": "PCIe transmit spread",
    "mem_util_per_gpu_util_p95": "mem / GPU util",
    "first_30s_mem_ratio": "startup peak / start",
    "mem_clock_mhz_p50": "memory clock",
    "acf10_power": "power correlation",
}


def _cut_text(feature: str, thr: float) -> str:
    if "kurt" in feature:
        return f"{thr:.1f}"
    if feature.startswith("mem_used_mb"):
        return f"{thr / 1024:.1f} GB"
    if feature == "first_30s_mem_ratio":
        return f"{thr:.1f}×"
    if "pct" in feature:
        return f"{thr:.1f} pts"
    if "temperature" in feature:
        return f"{thr:.0f} C"
    if "clock" in feature:
        return f"{thr:.0f} MHz"
    if "mbps" in feature:
        return f"{thr:.0f} MB/s"
    if feature == "acf10_power":
        return f"{thr:.2f}"
    return f"{thr:.2g}"


def _val_text(feature: str, val: float) -> str:
    if "kurt" in feature:
        return f"{val:.1f}"
    if feature.startswith("mem_used_mb"):
        return f"{val / 1024:.0f} GB"
    if feature == "first_30s_mem_ratio":
        return f"{val:.0f}×"
    if "pct" in feature:
        return f"{val:.1f}"
    if "temperature" in feature:
        return f"{val:.0f} C"
    if "clock" in feature:
        return f"{val:.0f}"
    if "mbps" in feature:
        return f"{val:.0f}"
    return f"{val:.2g}"


def _tree_nodes(model, tree_index: int, max_depth: int | None, expect_root: str):
    cols = list(pd.read_parquet(HERE / "frozen" / "gpu_cols.parquet").col)
    t = model.estimators_[tree_index].tree_
    root = cols[int(t.feature[0])]
    if root != expect_root:
        raise RuntimeError(f"tree {tree_index} root is {root}, expected {expect_root}")
    nodes = {}

    def walk(i, depth):
        f = int(t.feature[i])
        v = np.asarray(t.value[i]).reshape(-1)
        node = dict(
            id=i, depth=depth, leaf=f < 0,
            feature=None if f < 0 else cols[f],
            thr=None if f < 0 else float(t.threshold[i]),
            left=None if f < 0 else int(t.children_left[i]),
            right=None if f < 0 else int(t.children_right[i]),
            p=float(v[1] / v.sum()), n=int(t.n_node_samples[i]),
        )
        nodes[i] = node
        if f >= 0 and (max_depth is None or depth < max_depth):
            walk(node["left"], depth + 1)
            walk(node["right"], depth + 1)
    walk(0, 0)
    return nodes


def _follow(nodes, values):
    """Node ids from the root, following values. Stops at a leaf or at a missing value."""
    path = [0]
    while not nodes[path[-1]]["leaf"]:
        n = nodes[path[-1]]
        if n["feature"] not in values:
            break
        child = n["left"] if values[n["feature"]] <= n["thr"] else n["right"]
        if child not in nodes:
            path.append(child)  # child exists in the real tree but was not drawn
            break
        path.append(child)
    return path


def _layout(nodes):
    xs = {}
    cursor = [0.0]

    def place(i):
        n = nodes[i]
        kids = [c for c in (n["left"], n["right"]) if c in nodes]
        if not kids:
            xs[i] = cursor[0]
            cursor[0] += 1.0
            return
        for c in kids:
            place(c)
        xs[i] = 0.5 * (xs[kids[0]] + xs[kids[-1]])
    place(0)
    return xs


def _draw_tree(nodes, path, values, path_out, stub_at, figsize):
    """stub_at: node id -> extra line, for a path that continues below the drawing."""
    xs = _layout(nodes)
    depths = [n["depth"] for n in nodes.values()]
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    ax.set_xlim(min(xs.values()) - 0.62, max(xs.values()) + 0.62)
    ax.set_ylim(max(depths) + (1.25 if stub_at else 0.55), -0.55)
    ax.axis("off")
    path_set = set(path)

    def center(i):
        return xs[i], nodes[i]["depth"]

    for i, n in nodes.items():
        for c, word in ((n["left"], "below"), (n["right"], "above")):
            if c not in nodes:
                continue
            on = i in path_set and c in path_set
            ax.plot([xs[i], xs[c]], [n["depth"], nodes[c]["depth"]],
                    color=STAGE if on else "#D9DEE5", lw=2.0 if on else 0.7,
                    solid_capstyle="round", zorder=1)
            if on:
                mx = 0.55 * xs[i] + 0.45 * xs[c]
                my = 0.55 * n["depth"] + 0.45 * nodes[c]["depth"]
                ax.text(mx, my, word, color=STAGE, fontsize=6.5,
                        ha="center", va="center", zorder=2,
                        bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none"))

    for i, n in nodes.items():
        on = i in path_set
        if n["leaf"]:
            train = n["p"] >= 0.5
            face = "#1F7A4D" if on and train else ("#E7F6EE" if train else "#F3F4F6")
            edge = "#1F7A4D" if train else "#9AA5B1"
            ink = "white" if on and train else (GREEN if train else MUTED)
            label = "training" if train else "not training"
            text = f"{label}\n{n['n']} windows"
        else:
            face = "#1B4F72" if on else "white"
            edge = "#1B4F72"
            ink = "white" if on else INK
            name = SHORT.get(n["feature"], n["feature"])
            text = f"{name}\n≤ {_cut_text(n['feature'], n['thr'])}"
            if on and n["feature"] in values:
                text += f"\nwindow {_val_text(n['feature'], values[n['feature']])}"
        ax.text(xs[i], n["depth"], text, ha="center", va="center", fontsize=6.2,
                color=ink, zorder=3, linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.28", fc=face, ec=edge, lw=0.8))
        if i in stub_at:
            ax.plot([xs[i], xs[i]], [n["depth"] + 0.34, n["depth"] + 0.52],
                    color=STAGE, lw=1.3, zorder=2)
            ax.text(xs[i], n["depth"] + 0.56, stub_at[i], ha="center", va="top",
                    fontsize=6.5, color=STAGE, zorder=3)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_stage_tree(model, path: Path):
    nodes = _tree_nodes(model, STAGE_WALK["tree"], None, STAGE_WALK["root"])
    taken = _follow(nodes, STAGE_WALK["values"])
    if nodes[taken[-1]]["leaf"] is not True or nodes[taken[-1]]["p"] < 0.5:
        raise RuntimeError(f"stage walk ended unexpectedly at {taken[-1]}")
    _draw_tree(nodes, taken, STAGE_WALK["values"], path, {}, (7.3, 4.6))
    return taken


def fig_fallback_top(model, path: Path):
    # Three levels only. The example continues past the drawing.
    nodes = _tree_nodes(model, FALL_WALK["tree"], 2, FALL_WALK["root"])
    full = _tree_nodes(model, FALL_WALK["tree"], None, FALL_WALK["root"])
    taken_full = _follow(full, FALL_WALK["values"])
    taken = [i for i in taken_full if i in nodes]
    extra = len(taken_full) - len(taken)
    stub = {}
    if extra > 0:
        stub[taken[-1]] = f"{extra} more questions\nthen a training leaf"
    _draw_tree(nodes, taken, FALL_WALK["values"], path, stub, (7.3, 3.6))
    return taken_full


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
    ax.set_xlim(0, max(12, max(top.impurity.values) * 100 * 1.55))
    for i, (imp, roots) in enumerate(zip(top.impurity, top.n_roots)):
        ax.text(imp * 100 + 0.25, i, f"{int(roots)} trees ask this first",
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
    s["q"] = ParagraphStyle("q", fontName="Helvetica", fontSize=8, leading=10.6,
                             textColor=HexColor(INK))
    s["qh"] = ParagraphStyle("qh", fontName="Helvetica-Bold", fontSize=8, leading=10.6,
                              textColor=HexColor(INK))
    s["foot"] = ParagraphStyle("foot", fontName="Helvetica", fontSize=8, leading=10.5,
                                textColor=HexColor(MUTED))
    s["formula"] = ParagraphStyle("formula", fontName="Courier", fontSize=9, leading=12,
                                   textColor=HexColor(INK), alignment=1)
    return s


def _p(text, style):
    return Paragraph(text, style)


def _feat(frame, name):
    return frame.loc[frame.feature == name].iloc[0]


def _where(row) -> str:
    n = int(row.n_roots)
    frac = float(row.root_frac)
    depth = float(row.mean_depth)
    if n == 0:
        return f"Never the first question. It shows up about {depth:.0f} steps down."
    head = f"First question in {n} of 200 trees."
    if frac >= 0.35:
        return head + (f" {frac * 100:.0f}% of the mix this column removes "
                       "comes from that first question.")
    if depth < 2.5:
        return head + " Most of the mix it removes comes from one or two questions later."
    return head + f" Most of the mix it removes is about {depth:.0f} steps down."


def _above(row) -> str:
    g = float(row.vote_gap)
    if g >= 0.8:
        return (f"Training. Above the cut, the training share is {g:.2f} higher "
                "than below it, so the two sides are nearly pure.")
    if g >= 0.25:
        return (f"Training. Above the cut, the training share is {g:.2f} higher "
                "than below it.")
    if g <= -0.25:
        return (f"Not training. Above the cut, the training share is {abs(g):.2f} lower "
                "than below it.")
    return f"The two sides barely differ. The training share differs by {g:+.2f}."


def _qtable(st, rows):
    data = [[_p(h, st["qh"]) for h in ("The question", "Where in the tree", "Above the cut")]]
    for question, where, above in rows:
        data.append([_p(question, st["q"]), _p(where, st["q"]), _p(above, st["q"])])
    tbl = Table(data, colWidths=[2.55 * inch, 2.35 * inch, 2.2 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(PALE)),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


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
    story.append(_p(
        "Stage 2 asks whether memory utilization or power is moving. If it is, the window is "
        "called training. Fallback asks whether memory jumped in the first 30 samples of the "
        "run. If it did, the window is called training. Two of fallback's later questions point "
        "the other way: a GPU that is nearly full, and a memory bus that is busy while the "
        "processors are not. The tables list each question, where it sits in the tree, and "
        "which side is called training.",
        st["body"]))
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
        f"drawn at each split, class weights balanced, seed 0. The stage 2 and fallback "
        f"sections list the questions those trees ask. A cut quoted there is the typical "
        f"threshold, the median "
        f"of many split points weighted by how much mix each split removed. "
        f"\"Where the impurity goes\" says how the bar chart counts those questions.",
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

    story.append(CondPageBreak(3.2 * inch))
    story.append(_p("Where the impurity goes", st["h2"]))
    story.append(_p(
        "Skip ahead to Stage 2 if you want the questions first. This section is only how "
        "the bar lengths were counted.",
        st["body"]))
    story.append(_p(
        "Each tree is a stack of yes/no questions. A question looks at one column "
        "(\"is the spread of memory utilization above 1.2?\") and sends the window left or right.",
        st["body"]))
    story.append(_p(
        "Before the question, training and not-training are mixed together. Gini measures that "
        "mix. With two classes it is 2p(1-p), where p is the share of the weight that is "
        "training. It equals 0.5 when the two classes are even, and 0 when every window on "
        "that side is the same class. A useful question is one that makes the mix drop. The "
        "drop at a node is",
        st["body"]))
    formula = Table(
        [[_p("N * Gini  -  N_left * Gini_left  -  N_right * Gini_right", st["formula"])]],
        colWidths=[7.1 * inch],
    )
    formula.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(PALE)),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.4, HexColor(LINE)),
    ]))
    story.append(formula)
    story.append(Spacer(1, 6))
    story.append(_p(
        "N is not a count of windows. The forest was fit with balanced class weights, so the "
        "scarcer class pulls harder than its share of the rows, and N is that weight, already "
        "stored on the node. We add the drop over every question in all 200 trees and give the "
        "sum to the column the question used. Divide by the total drop. That percentage is the "
        f"bar. It is the same number sklearn calls feature importance. Recomputing it from the "
        f"nodes matches sklearn to {max(sm['importance_gap'], fm['importance_gap']):.4f}.",
        st["body"]))
    story.append(_p(
        "The bars are not 174 separate columns. The spread, the range, and the interquartile "
        "range of memory utilization are one measurement saved three times. Drawn separately, "
        "one fact would look like three. Columns that come from the same place are added:",
        st["body"]))
    groups = [
        ["Bar", "Which columns go into it"],
        ["Memory", "The name contains \"mem\": utilization, megabytes in use, and the jump in the first 30 samples."],
        ["Power and temperature", "The name contains \"power\" or \"temperature\"."],
        ["PCIe", "The name starts with \"pcie\"."],
        ["Compute", "Whatever is left: GPU utilization, SM clock, idle fraction, tensor activity."],
        ["Video", "Encoder and decoder. Those columns do not vary here, so the share is zero."],
    ]
    gdata = [[_p(groups[0][0], st["cellh"]), _p(groups[0][1], st["cellh"])]]
    for name, desc in groups[1:]:
        gdata.append([_p(name, st["cell"]), _p(desc, st["cell"])])
    gtbl = Table(gdata, colWidths=[1.7 * inch, 5.4 * inch])
    gtbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(PALE)),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(gtbl)
    story.append(Spacer(1, 6))
    story.append(_p(
        f"A column scores only at questions where the tree actually picked it. Each question "
        f"is shown {sm['max_features']} of the {sm['n_features']} columns, so a twin that was "
        f"not shown gets nothing for that question. And stage 2's memory bar of "
        f"{sg.get('Memory', 0) * 100:.0f}% does not mean memory caused the call. The startup "
        f"jump and the utilization wobble are both inside that bar. Figure 2 is the total. "
        f"The two tables below split that bar into the actual questions.",
        st["body"]))
    story.append(KeepTogether([
        _img(paths["groups"]),
        _p(
            "Figure 2. Of all the mix a forest removed, the share removed by questions about "
            f"each source. Stage 2's memory bar is {sg.get('Memory', 0) * 100:.0f}%: that share "
            "of the mix it removed came from a split on a column whose name contains \"mem\".",
            st["cap"]),
    ]))
    s_pow_share = float(sg.get("Power & thermal", 0)) * 100
    f_pow_share = float(fg.get("Power & thermal", 0)) * 100
    story.append(KeepTogether([
        _p(
            f"Stage 2 puts {sg.get('Memory', 0) * 100:.0f}% on memory and "
            f"{s_pow_share:.0f}% on power and temperature. "
            f"PCIe is {sg.get('PCIe', 0) * 100:.0f}%. Compute, which is where GPU utilization "
            f"and SM clock live, is {sg.get('Compute', 0) * 100:.0f}%, and none of that is at the "
            f"root. GPU-utilization mean is rank "
            f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'gpu_utilization_pct_mean'].iloc[0])} "
            f"and removes essentially no impurity.",
            st["body"]),
        _p(
            f"Fallback is still {fg.get('Memory', 0) * 100:.0f}% memory, but the memory it uses "
            f"is the allocation at the start of the run. Compute rises to "
            f"{fg.get('Compute', 0) * 100:.0f}%, PCIe to {fg.get('PCIe', 0) * 100:.0f}%, and power "
            f"falls to {f_pow_share:.0f}%.",
            st["body"]),
    ]))

    story.append(_p("Stage 2: is memory or power moving?", st["h2"]))
    story.append(_p(
        f"Stage 2 trained only on windows where the network rule had already fired: "
        f"3,019 training windows and 939 others. The forest is short, average depth "
        f"{sm['mean_depth']:.1f}, about {sm['mean_leaves']:.0f} leaves per tree. "
        f"The not-training windows in that fit are HPC jobs with balanced traffic and a flat "
        f"GPU trace. Inference was not in the fit. A serving window whose memory utilization "
        f"moves lands on the training side of every question below, which is why stage 2 "
        f"never grew a leaf that recognizes inference.",
        st["body"]))
    s_iqr = _feat(stage, "mem_utilization_pct_iqr")
    story.append(_qtable(st, [
        (f"Is the spread of memory utilization above {s_std.threshold:.1f} percentage points?",
         _where(s_std), _above(s_std)),
        (f"Is the range of that same utilization above { _feat(stage, 'mem_utilization_pct_range').threshold:.1f} "
         f"percentage points? Same measurement, stored again.",
         _where(_feat(stage, "mem_utilization_pct_range")),
         _above(_feat(stage, "mem_utilization_pct_range"))),
        (f"Is the interquartile range of that same utilization above {s_iqr.threshold:.1f} "
         f"percentage points? Same measurement, a third copy.",
         _where(s_iqr), _above(s_iqr)),
        (f"Is the spread of power above {s_pow.threshold:.0f} W?",
         _where(s_pow), _above(s_pow)),
        (f"Is GPU memory in use above {s_mem.threshold / 1024:.0f} GB?",
         _where(s_mem), _above(s_mem)),
        (f"Does memory take longer than {s_plat.threshold:.0f} samples to reach its plateau? "
         f"A slow climb is above this cut.",
         _where(s_plat), _above(s_plat)),
    ]))
    story.append(_p(
        "Table 1. Stage 2's questions. \"Above the cut\" is the side where the measured value "
        "is higher. The training-share difference is the tree's own class mix on the high side "
        "minus the mix on the low side, weighted by how much mix the split removed. "
        "A difference of 1 would mean one side is all training and the other side is all "
        "not-training. Depth 0 is the first question in a tree.",
        st["cap"]))
    story.append(_p(
        f"The startup jump that dominates fallback ranks between "
        f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'first_30s_mem_ratio'].iloc[0])} "
        f"and "
        f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'first_30s_mem_delta_mb'].iloc[0])} "
        f"here, so stage 2 barely asks it. GPU-utilization mean ranks "
        f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'gpu_utilization_pct_mean'].iloc[0])}. "
        f"Spread, range, and interquartile range together are the first question in "
        f"{int(spread.n_roots.sum())} of the 200 trees. Each split is shown only "
        f"{sm['max_features']} of the {sm['n_features']} columns, so a measurement that is the "
        f"first question about as often as it is offered is one the tree actually prefers.",
        st["body"]))
    story.append(_img(paths["stage"]))
    story.append(_p(
        "Figure 3. The same stage 2 columns, sized by how much mix they remove. "
        "Green means above the cut is training. Several bars are one measurement: "
        "spread, range, and the interquartile range of memory utilization.",
        st["cap"]))

    story.append(CondPageBreak(3.4 * inch))
    story.append(_p("Fallback: did memory jump at startup?", st["h2"]))
    story.append(_p(
        f"Fallback trained on every honest window. It is a deeper forest, average depth "
        f"{fm['mean_depth']:.1f}, about {fm['mean_leaves']:.0f} leaves per tree. "
        f"Its first questions read the first 30 samples of the run, and only as far as the "
        f"current window is allowed to see. Once those 30 samples have passed, later windows "
        f"of the same job all see the same jump. The three startup rows below are that one "
        f"jump written three ways. Together they are {startup.impurity.sum() * 100:.0f}% of "
        f"the mix fallback removes, and {startup.root_share.sum() * 100:.0f}% of the weight "
        f"of first questions. The code comment says the feature was written to catch optimizer "
        f"state allocated at startup, on the order of 6 GB.",
        st["body"]))
    f_pcie = _feat(fall, "pcie_tx_mbps_range")
    story.append(_qtable(st, [
        (f"Did memory rise by more than {f_d.threshold / 1024:.1f} GB in those first 30 samples?",
         _where(f_d), _above(f_d)),
        (f"Is the peak in those samples more than {f_r.threshold:.0f} times the starting value? "
         f"Same jump, as a ratio.",
         _where(f_r), _above(f_r)),
        (f"Is that rise more than {f_f.threshold * 100:.0f}% of the GPU's memory? "
         f"Same jump, as a fraction of capacity.",
         _where(f_f), _above(f_f)),
        (f"Is memory utilization minus GPU utilization above {f_diff.threshold:.0f} points? "
         f"Above this, the memory bus is busy relative to the processors.",
         _where(f_diff), _above(f_diff)),
        (f"Is GPU memory in use above {f_mem.threshold / 1024:.0f} GB? "
         f"Stage 2's cut on this same column is {s_mem.threshold / 1024:.0f} GB, and there the high side is training.",
         _where(f_mem), _above(f_mem)),
        (f"Is the range of PCIe transmit above {f_pcie.threshold:.0f} MB/s?",
         _where(f_pcie), _above(f_pcie)),
    ]))
    story.append(_p(
        "Table 2. Fallback's questions, read the same way as Table 1. "
        "The fourth and fifth rows point the other way: above the cut is not training.",
        st["cap"]))
    story.append(_img(paths["fall"]))
    story.append(_p(
        "Figure 4. The same fallback columns as Table 2, sized by how much mix they remove. "
        "Green: above the cut is training. Red: above the cut is not training. "
        "Gray: the two sides barely differ. The first three bars are one jump, written three ways.",
        st["cap"]))
    story.append(KeepTogether([
        _img(paths["foot"]),
        _p(
            "Figure 5. One column from Table 2, the two forests disagree on it. "
            f"Stage 2 calls training above about {s_mem.threshold / 1024:.0f} GB. "
            f"Fallback calls not-training above about {f_mem.threshold / 1024:.0f} GB. "
            "Both cuts are weighted medians of split points.",
            st["cap"]),
    ]))
    story.append(_p(
        f"GPU utilization is not one of these questions. In fallback its mean ranks "
        f"{int(fall.impurity.rank(ascending=False)[fall.feature == 'gpu_utilization_pct_mean'].iloc[0])}, "
        f"and the typical cut is {f_util.threshold:.1f}%, so only a pinned GPU registers. "
        f"The training share above that cut is {f_util.vote_gap:.2f} higher than below it, "
        f"a weak split. In stage 2 this column ranks "
        f"{int(stage.impurity.rank(ascending=False)[stage.feature == 'gpu_utilization_pct_mean'].iloc[0])}.",
        st["body"]))

    story.append(_p("One stage 2 tree, and one window walking through it", st["h2"]))
    story.append(_p(
        "Stage 2 is 200 trees, not one tree. "
        f"The drawing is tree {STAGE_WALK['tree']}, one of the 17 whose first question is the "
        f"spread of memory utilization. It has 10 leaves. A number of windows on a leaf is how "
        f"many training-fit windows reached it. The blue path is one 30-second window of "
        f"{STAGE_WALK['cell']}. Its spread is "
        f"{STAGE_WALK['values']['mem_utilization_pct_std']:.1f} points, its memory kurtosis is "
        f"{STAGE_WALK['values']['mem_used_mb_kurt']:.0f}, and its memory in use is "
        f"{STAGE_WALK['values']['mem_used_mb_max'] / 1024:.0f} GB. The rule has fired on this "
        f"window: about {STAGE_WALK['mibps']:.0f} MiB/s, symmetry {STAGE_WALK['sym']:.2f}.",
        st["body"]))
    story.append(_img(paths["stage_tree"]))
    story.append(_p(
        "Figure 6. Every node of stage 2 tree "
        f"{STAGE_WALK['tree']}. Below means the measured value is under the cut, above means "
        "it is over the cut. The window takes the blue path and ends in the training leaf "
        "that holds most of the windows this tree saw. The cuts on this one tree are not the "
        "typical cuts in Table 1. This tree asks about memory in use at 5.8 GB. The typical "
        "cut across all 200 trees is about 10 GB.",
        st["cap"]))
    story.append(_p(
        f"The kurtosis question peels off only a few dozen windows. Almost every window with "
        f"spread above 1.2 points continues, and a memory footprint above 5.8 GB is then called "
        f"training. On this window, {STAGE_WALK['votes']} of the 200 trees vote training "
        f"(probability {STAGE_WALK['proba']:.2f}). Fallback's probability on the same window "
        f"is {STAGE_WALK['other_proba']:.2f}, and it is not asked, because the rule fired.",
        st["body"]))

    story.append(_p("One fallback tree, the first three levels", st["h2"]))
    story.append(_p(
        f"A fallback tree has about {fm['mean_leaves']:.0f} leaves, so the whole tree does not "
        f"fit on a page. This is tree {FALL_WALK['tree']}, one of the 15 whose first question "
        f"is the startup peak divided by the start. Only the first three levels are drawn. "
        f"The blue path is one 30-second window of {FALL_WALK['cell']}. The ratio is "
        f"{FALL_WALK['values']['first_30s_mem_ratio']:.0f}, because the feature floors the "
        f"starting memory at 1 MB and this run's first samples are near empty while the peak "
        f"is about 10 GB. The rule is silent: about {FALL_WALK['mibps']:.0f} MiB/s on the wire, "
        f"so this forest is the one that gets asked.",
        st["body"]))
    story.append(_img(paths["fall_tree"]))
    story.append(_p(
        "Figure 7. The top of fallback tree "
        f"{FALL_WALK['tree']}. The window goes above the startup cut, then below the PCIe cut, "
        "then through three more questions (memory clock, then temperature twice) and reaches "
        "a leaf that says training. Those three questions are the kind of fine split a deep "
        "tree grows. They are not the questions the forest leads with.",
        st["cap"]))
    story.append(_p(
        f"On this window, {FALL_WALK['votes']} of the 200 fallback trees vote training "
        f"(probability {FALL_WALK['proba']:.2f}). One tree reaching a training leaf is not the "
        f"forest's vote. The vote is how many of the 200 agree.",
        st["body"]))

    story.append(KeepTogether([
        _p("What that does to the gate", st["h2"]),
        _p(
            "On the 93 Nebius jobs already scored with this frozen detector, the pattern matches "
            "the cuts. Those jobs were not used to choose features or to move the 0.5 cutoff. "
            "Probabilities below are the mean over the job's windows.",
            st["body"]),
    ]))

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
        "Table 3. Orange rows are decided by fallback, because the rule is silent. Blue rows "
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

    story.append(KeepTogether([
        _p("Cells these cuts have not seen", st["h2"]),
        _p(
            "Designs, not scores. None of these jobs has been captured. They are not a reason "
            "to move the 0.5 cutoff or to refit either forest. Do not add them to a training "
            "set and then score them: a tree that has seen the cell can memorize it. "
            "The point of running each one once, against this freeze, is to see which cut decides.",
            st["body"]),
    ]))
    story.append(_p(
        "The startup feature reads the first 30 samples of the capture, about 30 seconds, and "
        "every later window reuses that same prefix. A jump after sample 30 never enters the "
        "feature. Sleeping 40 seconds and then allocating does hide the jump. It also sets "
        "time-to-plateau near 40 samples. Fallback cuts that column at 16 samples, and the "
        "slow side is training (7 of 200 trees ask it first; the training share above the cut "
        "is 0.38 higher than below it). That is a weaker question than the jump, and it points "
        "the same way. A sampler that starts only after memory is already flat hides both: "
        "there is no jump, and time-to-plateau is 0 because the memory is already at its plateau. "
        "Those are different cells.",
        st["body"]))
    story.append(_p(
        "One more fallback question still points at a real training step even when both of "
        "those are hidden. Memory utilization minus GPU utilization is cut at about -23 points. "
        "The high side is not training. A step that pins the processors and barely touches "
        "the memory bus sits on the training side. A model that fills the GPU does not: "
        "memory utilization is high too, so the difference is the other side of -23. "
        "The 68 GB footprint cut is the same shape. Above it, fallback votes not-training. "
        "A small model is on the training side of that cut. A held buffer between 68 and 78 GB "
        "is the single change that crosses it.",
        st["body"]))

    design_header = ["Cell", "Asked", "Do this", "Called training", "Called not training"]
    design_rows = [
        ["Warm attach, large",
         "fallback",
         "InfiniBand off. Start the sampler only after memory has been flat for a minute. "
         "Hold one extra buffer so memory stays between 68 and 78 GB.",
         "The jump was not what caught the orange rows. Name the column that still votes training.",
         "Once the jump is gone, the full-GPU cut is enough. This is the sharpest miss the cuts allow."],
        ["Warm attach, small",
         "fallback",
         "Same late sampler, same quiet wire. Plateau under 10 GB. No held buffer. Compare only to the large row.",
         "A small footprint is still enough for fallback, or the processors-versus-memory question is.",
         "Hiding the jump is enough even under 10 GB. The 68 GB cut was not required."],
        ["Sleep, then allocate",
         "fallback",
         "Sampler already running. Sleep 40 seconds, then allocate the same small model. InfiniBand off.",
         "The slow plateau replaced the jump. This row is not a clean removal of the startup tell.",
         "Hiding the jump is enough even when time-to-plateau is on the training side."],
        ["Flat counters, real all-reduce",
         "stage 2",
         "A real small step. Send+receive at least 1 MiB/s, symmetry at least 0.8. "
         "Memory under 10 GB, utilization range under 4.2 points (about 3.4 GB on an 80 GB GPU), "
         "power spread under 9 W, plateau inside 16 samples. Fixed clocks, no checkpoint, no sleep.",
         "Either a Table 1 cut was actually crossed, or some other column is carrying the vote. The trace has to say which.",
         "The first stage 2 veto in this score. The 1,032 rule-on windows never went below 0.62."],
        ["Ordinary small step",
         "stage 2",
         "Same all-reduce and the same small model. Ordinary clocks, ordinary allocator. Do not hold the counters flat.",
         "A real step does not live on the not-training side of Table 1. Record which cut it crossed.",
         "Small training already evades, with no costume."],
        ["Empty wobble",
         "stage 2",
         "No model. Hold about 12 GB. In every window, move memory utilization by more than 4.2 points, about 3.4 GB on an 80 GB GPU. "
         "Balanced all-reduce of a few MiB/s for the whole capture, not one burst.",
         "The rule, the 10 GB footprint, and the wobble are sufficient. The blue rows in Table 3 are this shape.",
         "Those jobs are tripping a column that is not in Table 1."],
        ["Weight load, quiet wire",
         "fallback",
         "Capture from process start. Load weights in the first 30 seconds. Stay under 68 GB. No cross-node traffic.",
         "A weight load and the optimizer jump look the same to fallback.",
         "Fallback already separates them. The 93 jobs do not show it: their inference false alarms all have the rule on, and fallback was ignored."],
        ["Weight load past 68 GB",
         "fallback",
         "Same weight load, same quiet wire. Plateau between 68 and 78 GB, and nothing else changes.",
         "The jump wins over the footprint cut.",
         "The footprint cut wins. The two questions in Table 2 that point opposite ways have been put on one job."],
    ]
    design = [[_p(h, st["cellh"]) for h in design_header]]
    for r in design_rows:
        design.append([_p(c, st["cell"]) for c in r])
    design_w = [1.15 * inch, 0.62 * inch, 2.15 * inch, 1.64 * inch, 1.64 * inch]
    design_tbl = Table(design, colWidths=design_w, repeatRows=1)
    design_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(PALE)),
        ("BACKGROUND", (0, 1), (-1, 5), HexColor("#F8EFEA")),
        ("BACKGROUND", (0, 6), (-1, 8), HexColor("#E7F0F7")),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(design_tbl)
    story.append(_p(
        "Table 4. Orange rows are training. Blue rows are not. "
        "\"Asked\" is the forest that speaks if the wire does what the row says. "
        "The last two columns are how to read the call. They are not predictions.",
        st["cap"]))
    story.append(KeepTogether([
        _p(
            "Asymmetric traffic with the jump left in the first 30 samples is a control, not another "
            "evasion. Symmetry under 0.8 keeps the rule silent, and fallback should still call the "
            "job training, the way it called iboff_flat. If it does not, the jump was not the reason "
            "those orange rows were caught.",
            st["body"]),
        _p(
            "qwen_fake_ar, olmoe_ep_mixed, and disagg_echo are already the blue rows of Table 3. "
            "qwen_one_spike is already a single burst, and only about a tenth of its windows were "
            "called training. iboff_flat, nosync_flat, and lora_1n8_iboff are already a visible jump "
            "on a quiet wire. None of the rows above is a rerun of those.",
            st["body"]),
    ]))

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
        "stage_tree": FIG / "stage_tree.png",
        "fall_tree": FIG / "fall_tree.png",
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
    fig_stage_tree(stage2, paths["stage_tree"])
    fig_fallback_top(fallback, paths["fall_tree"])
    pdf_path = OUT / "frozen_forest_interp.pdf"
    build_pdf(stage, fall, sm, fm, paths, pdf_path)
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")
    print("stage2", sm)
    print("fallback", fm)


if __name__ == "__main__":
    main()
