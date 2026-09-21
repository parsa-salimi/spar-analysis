"""Enumerate a grid of realistic configurations and measure how separable the
two classes are from interconnect traffic alone.

IMPORTANT CAVEAT, stated up front because it governs how the numbers may be
read: this is an ANALYTIC grid, not a sample of the world. Every AUC below is a
property of the configuration prior encoded in the grid -- how many training
configs, how many inference configs, at what scales -- and is NOT an empirical
detection rate. It answers "could a perfect observer of byte volume tell these
apart?", which is a feasibility bound. It does not answer "what will our
detector score?", which needs real traces.
"""
from __future__ import annotations
import itertools, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import commvol as C

HWS  = ["A100-80GB", "H100-SXM", "H200-SXM", "B200"]
MODS = list(C.MODELS)

def parallel_combos(worlds=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    out = []
    for w in worlds:
        for tp in (1, 2, 4, 8):
            for pp in (1, 2, 4):
                if tp * pp > w:                continue
                if w % (tp * pp):              continue
                dp = w // (tp * pp)
                if tp > 8:                     continue      # TP must stay in a node
                out.append((dp, tp, pp))
    return sorted(set(out))

def build() -> pd.DataFrame:
    rows = []
    combos = parallel_combos()
    for mn, hn, (dp, tp, pp), gb, seq, ga, ckpt, sh in itertools.product(
            MODS, HWS, combos, (8, 64, 512, 2048), (2048, 8192), (1, 4),
            (False, True), ("ddp", "fsdp")):
        m, hw = C.MODELS[mn], C.HARDWARE[hn]
        if tp > m.n_heads:                 continue
        if pp > m.layers:                  continue
        w = C.training_comm(m, hw, dp=dp, tp=tp, pp=pp, global_batch=gb, seq=seq,
                            sharding=sh, grad_accum=ga, checkpointing=ckpt)
        if w.feasible:
            rows.append(w.as_row())
    for mn, hn, mode, tp, pp, rep, bs, pl, ol in itertools.product(
            MODS, HWS, ("colocated", "disaggregated", "replicated"),
            (1, 2, 4, 8), (1, 2, 4), (1, 4, 16), (16, 64, 256),
            (512, 2048, 8192), (64, 256, 1024)):
        m, hw = C.MODELS[mn], C.HARDWARE[hn]
        if tp > m.n_heads:                  continue
        if pp > m.layers:                   continue
        if mode == "replicated" and (tp > 1 or pp > 1): continue
        w = C.inference_comm(m, hw, mode=mode, tp=tp, pp=pp, replicas=rep, batch=bs,
                             prompt_len=pl, output_len=ol)
        if w.feasible:
            rows.append(w.as_row())
    return pd.DataFrame(rows)


def auc_1d(df, col, invert=False):
    x = np.log10(np.maximum(df[col].values, 1.0))
    y = (df.cls.values == "training").astype(int)
    if len(np.unique(y)) < 2: return np.nan
    a = roc_auc_score(y, -x if invert else x)
    return max(a, 1 - a)            # a threshold rule may point either way


def held_out_model_auc(df, cols):
    """Train on 8 model families, score the 9th. Mirrors the leave-one-family-out
    discipline from the power analysis: no memorising a specific model."""
    y = (df.cls.values == "training").astype(int)
    X = df[cols].values
    out = {}
    for mo in sorted(df.model.unique()):
        te = df.model.values == mo
        if len(np.unique(y[~te])) < 2 or len(np.unique(y[te])) < 2: continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[~te], y[~te])
        out[mo] = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
    return out


if __name__ == "__main__":
    df = build()
    df["log_total"] = np.log10(np.maximum(df.total_bps, 1))
    df["log_inter"] = np.log10(np.maximum(df.inter_bps, 1))
    df["log_intra"] = np.log10(np.maximum(df.intra_bps, 1))
    df["log_msg"]   = np.log10(np.maximum(df.msg_bytes, 1))
    df.to_csv("grid.csv.gz", index=False, compression="gzip")
    print(f"configurations: {len(df):,}  "
          f"(training {int((df.cls=='training').sum()):,}, "
          f"inference {int((df.cls=='inference').sum()):,})")

    print("\n== per-GPU interconnect throughput, GB/s ==")
    q = df.groupby("cls").total_bps.describe(percentiles=[.05,.25,.5,.75,.95])
    print((q[["5%","25%","50%","75%","95%"]]/1e9).to_string(float_format=lambda v:f"{v:,.1f}"))

    res = {}
    for name, col in [("total volume","total_bps"), ("inter-node (NIC) only","inter_bps"),
                      ("intra-node (NVLink) only","intra_bps")]:
        res[name] = auc_1d(df, col)
    print("\n== AUC from a single threshold on volume (direction free) ==")
    for k,v in res.items(): print(f"  {k:26s} {v:.3f}")
    print("\n== held-out-model AUC by feature tier (direction must be fixed in advance) ==")

    # ── what can a COUNTER-BASED verifier actually see? ─────────────────────
    # Our agents read cumulative byte counters (DCGM NVLink fields, sysfs
    # port_xmit_data) at a few Hz. That fixes which features are legitimate:
    #
    #  T1  rate per fabric, and tx/rx symmetry      -- separate counters exist
    #  T2  temporal structure of those rates        -- needs a fast sample rate
    #  T3  individual message sizes, collective count, collective type
    #                                               -- NOT VISIBLE from counters.
    #      Getting these needs an in-process hook, i.e. the adversary reporting
    #      on itself, which is exactly what a verifier cannot rely on.
    T1 = ["log_total", "log_inter", "log_intra", "sym"]
    T2 = T1 + ["interval_cv", "size_cv"]
    T3 = T2 + ["log_msg", "collectives_per_step"]
    VOL = ["log_total", "log_inter", "log_intra"]
    STRUCT = T2
    for name, cols in [("volume only (T1 minus symmetry)", VOL),
                       ("T1  counters + tx/rx symmetry", T1),
                       ("T2  + temporal structure", T2),
                       ("T3  + message size & count (NOT counter-observable)", T3)]:
        per = held_out_model_auc(df, cols)
        res[f"heldout_model::{name}"] = float(np.mean(list(per.values())))
        lo = min(per.values())
        print(f"  {name:52s} mean {np.mean(list(per.values())):.3f}   worst {lo:.3f}")

    # overlap: what fraction of training sits inside the inference IQR and back
    tr = df[df.cls=="training"].total_bps; inf = df[df.cls=="inference"].total_bps
    lo, hi = inf.quantile(.05), inf.quantile(.95)
    res["train_inside_inference_90pct_band"] = float(((tr>=lo)&(tr<=hi)).mean())
    lo2, hi2 = tr.quantile(.05), tr.quantile(.95)
    res["inference_inside_training_90pct_band"] = float(((inf>=lo2)&(inf<=hi2)).mean())
    print(f"\ntraining configs inside inference's 5-95% volume band: "
          f"{res['train_inside_inference_90pct_band']:.1%}")
    print(f"inference configs inside training's 5-95% volume band: "
          f"{res['inference_inside_training_90pct_band']:.1%}")
    # ── where does the "structure" AUC actually come from? ──────────────────
    print("\n== AUC broken out by inference mode (training as the positive class) ==")
    for mode in ["colocated", "disaggregated", "replicated"]:
        sub = df[(df.cls == "training") | (df.get("mode") == mode)]
        res[f"volume_vs_{mode}"] = auc_1d(sub, "total_bps")
        print(f"  training vs {mode:15s} n={len(sub):6,}  volume AUC {res[f'volume_vs_{mode}']:.3f}")

    print("\n== leave-one-feature-out on the structure model (held-out-model AUC) ==")
    mean_auc = lambda cols: float(np.mean(list(held_out_model_auc(df, cols).values())))
    b0 = mean_auc(["log_total"])
    print(f"  {'log_total alone':32s} {b0:.3f}")
    for extra in ["log_inter", "log_intra", "sym", "interval_cv", "size_cv",
                  "log_msg", "collectives_per_step"]:
        print(f"  + {extra:30s} {mean_auc(['log_total', extra]):.3f}")
    print(f"  {'all of them':32s} {mean_auc(STRUCT):.3f}")

    print("\n== the hard case: multi-GPU training vs COLOCATED inference only ==")
    hard = df[((df.cls=="training") & (df.world > 1)) | (df.get("mode")=="colocated")]
    for name, cols in [("volume only", VOL), ("T1", T1), ("T2", T2), ("T3", T3)]:
        m = float(np.mean(list(held_out_model_auc(hard, cols).values())))
        res[f"hard::{name}"] = m
        print(f"  {name:20s} {m:.3f}")

    cb = df.groupby("cls").comm_bound.mean()
    print(f"\ncommunication-bound configurations: "
          f"training {cb.get('training',0):.1%}, inference {cb.get('inference',0):.1%}")
    res["comm_bound_training"] = float(cb.get("training", 0))
    json.dump(res, open("separability.json","w"), indent=1)
