"""Measurement-artifact study: what does the INSTRUMENT cost us?

The agent has three settings Long has to choose: how often to read the counters,
how tightly to sync the clocks, and how wide the counters are. Each choice
discards information irreversibly. This measures how much detection each one
costs, so the schema can carry a requirement rather than a guess.

Why it must be synthetic: on real hardware you observe a workload THROUGH the
instrument, so poor detection could mean the adversary hid well or the sampling
was too coarse, and nothing separates those. Here the true bytes are generated
first and then degraded deliberately, one knob at a time, so the difference is
attributable by construction.

Design
------
* The traffic is built ONCE per (family, seed) and every condition observes the
  same events -- so a difference between conditions cannot be a different draw.
* Evaluation is leave-one-family-out, the protocol from the power analysis: one
  TRAINING family and one SERVING family are held out together (a single family
  would give a one-class test set and an undefined AUC). 7 x 4 = 28 pairs.
* Comparisons are PAIRED: same families, same seeds, same held-out pair, only
  the knob differs. With this few families an unpaired comparison of AUCs would
  be swamped by between-family variance -- the lesson from the power analysis.
* Controls: a CEILING (200 Hz, 64-bit, perfect clocks) that everything is
  expressed as a fraction of, and a FLOOR (labels shuffled) that must land at
  0.5 or the study leaks.

Usage:  python3 artifacts.py features <a> <b>   # cache features for conditions [a,b)
        python3 artifacts.py eval               # evaluate + write results
"""
from __future__ import annotations
import sys, json, time, itertools, shutil, tempfile
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/claude/commvol")
import commvol as C
from commtrace.gen import build_events, sample_events, Topology, Collection
from commtrace.reader import trace_features, FEATURE_COLS

OUT = Path(__file__).resolve().parent / "artifact_results"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "features"; CACHE.mkdir(exist_ok=True)

DURATION = 240.0
N_NODES = 4
SEEDS = (0, 1, 2)
WINDOW_S, STRIDE_S = 30.0, 15.0

H = C.HARDWARE["H100-SXM"]; M = C.MODELS

# ── the population: hard cases, not easy ones ───────────────────────────────
# Deliberately excludes replicated serving and single-GPU training: both emit
# nothing, are trivially separable from anything that does, and would inflate
# every number. The question here is how well the instrument resolves workloads
# that DO communicate.
# A FIRST ATTEMPT AT THIS POPULATION WAS USELESS and the failure is instructive.
# Every training family used tp=1 and every serving family tp=8, so no training
# family had ANY NVLink traffic and every serving family did -- "has NVLink
# traffic" separated the classes perfectly. Every serving family's inter-node
# traffic was also a directional KV push, so symmetry separated them perfectly
# too. Result: AUC 1.000 in every condition, no knob measurable, nothing learned.
#
# This population is built so the classes OVERLAP on the things that are easy to
# see. Both sides use tensor parallelism inside the node, so both have heavy
# NVLink all-reduce traffic. Both sides include pipeline-parallel configurations
# spanning nodes (p2p, which is symmetric at NODE level) and cross-node MoE
# expert parallelism (all-to-all, also symmetric). What is left to separate them
# is cadence and cross-node synchrony -- precisely the fragile quantities the
# instrument degrades, which is the point of the study.
CKPT = dict(sharding="fsdp", checkpointing=True)
TRAIN_FAMILIES = {
    "ddp_7b":       lambda: C.training_comm(M["Llama-2-7B"],  H, dp=32, tp=1, global_batch=128, seq=2048, **CKPT),
    "fsdp_tp_70b":  lambda: C.training_comm(M["Llama-3-70B"], H, dp=8,  tp=8, global_batch=64,  seq=2048, **CKPT),
    "fsdp_tp_32b":  lambda: C.training_comm(M["Qwen2.5-32B"], H, dp=8,  tp=8, global_batch=64,  seq=2048, **CKPT),
    "pp_train_70b": lambda: C.training_comm(M["Llama-3-70B"], H, dp=4,  tp=8, pp=2, global_batch=32, seq=2048, **CKPT),
    "moe_train":    lambda: C.training_comm(M["Mixtral-8x7B"],H, dp=8,  tp=8, global_batch=64,  seq=2048, **CKPT),
    "diloco_13b":   lambda: C.training_comm(M["Llama-2-13B"], H, dp=8,  tp=8, global_batch=64,  seq=2048,
                                            sharding="diloco", diloco_inner=100, checkpointing=True),
    "gradacc_32b":  lambda: C.training_comm(M["Qwen2.5-32B"], H, dp=8,  tp=8, global_batch=64,  seq=2048, grad_accum=8, **CKPT),
}
SERVE_FAMILIES = {
    "pp_serve_70b": lambda: C.inference_comm(M["Llama-3-70B"], H, mode="colocated", tp=8, pp=2, replicas=2, batch=64, prompt_len=2048),
    "pp_serve_32b": lambda: C.inference_comm(M["Qwen2.5-32B"], H, mode="colocated", tp=8, pp=4, replicas=1, batch=32, prompt_len=2048),
    "moe_serve":    lambda: C.inference_comm(M["Mixtral-8x7B"],H, mode="colocated", tp=8, pp=2, replicas=2, batch=64, prompt_len=2048),
    "disagg_70b":   lambda: C.inference_comm(M["Llama-3-70B"], H, mode="disaggregated", tp=8, replicas=4, batch=32, prompt_len=2048),
    "coloc_70b":    lambda: C.inference_comm(M["Llama-3-70B"], H, mode="colocated", tp=8, replicas=4, batch=64, prompt_len=2048),
}
FAMILIES = {**TRAIN_FAMILIES, **SERVE_FAMILIES}


@dataclass(frozen=True)
class Cond:
    name: str
    knob: str                 # which axis this row belongs to
    x: float                  # the knob's value, for plotting
    sample_hz: float = 10.0
    width: int = 64
    skew_ns: float = 1e6      # 1 ms: ordinary NTP on a LAN
    read_dur_ns: int = 80_000
    seed_offset: int = 0      # same traffic, different sampling randomness

    def collection(self, seed: int) -> Collection:
        return Collection(sample_hz=self.sample_hz,
                          ib_unit="words4", ib_width=self.width,
                          nvlink_unit="bytes", nvlink_width=self.width,
                          clock_skew_sd_ns=self.skew_ns,
                          read_dur_ns=self.read_dur_ns,
                          seed=seed + self.seed_offset)


def conditions() -> list[Cond]:
    # CEILING = the best instrument we would actually consider, not an infinitely
    # fast one. Sampling above ~100 Hz would resolve the artificial regularity
    # that MAX_EVENTS bundling introduces in fast flows (see commtrace/gen.py),
    # so a 200 Hz ceiling measures the generator's approximation rather than the
    # workload. 50 Hz is the top of the sweep and safely below that.
    cs = [Cond("ceiling", "control", np.nan, sample_hz=50.0, width=64,
               skew_ns=0.0, read_dur_ns=0),
          # NULL CONTROL: identical settings to the ceiling, same underlying
          # traffic, only the sampling randomness differs. Its paired difference
          # from the ceiling is by construction zero, so whatever spread it shows
          # is the experiment's own noise floor -- the bar every other condition
          # has to clear before it may be called an effect.
          Cond("null_dup", "control", np.nan, sample_hz=50.0, width=64,
               skew_ns=0.0, read_dur_ns=0, seed_offset=1000)]
    for hz in (0.2, 0.5, 1, 2, 5, 10, 20, 50):
        for w in (64, 32):
            cs.append(Cond(f"hz{hz}_w{w}", "sample_hz", hz, sample_hz=hz, width=w))
    for sk_ms in (0.0, 1.0, 10.0, 100.0, 1000.0):
        cs.append(Cond(f"skew{sk_ms}ms", "clock_skew_ms", sk_ms, skew_ns=sk_ms * 1e6))
    for rd_us in (0, 100, 1000, 10000):
        cs.append(Cond(f"read{rd_us}us", "read_skew_us", rd_us, read_dur_ns=rd_us * 1000))
    return cs


_EVENTS: dict = {}
def events_for(fam: str, seed: int):
    key = (fam, seed)
    if key not in _EVENTS:
        _EVENTS[key] = build_events(FAMILIES[fam](), duration_s=DURATION,
                                    topo=Topology(n_nodes=N_NODES), seed=seed)
    return _EVENTS[key]


def build_features(cond: Cond) -> pd.DataFrame:
    """Cache one condition's feature table: all families, all seeds."""
    f = CACHE / f"{cond.name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows, tmp = [], Path(tempfile.mkdtemp())
    try:
        for fam in FAMILIES:
            for seed in SEEDS:
                ev = events_for(fam, seed)
                p, _ = sample_events(ev, tmp / "t", coll=cond.collection(seed),
                                     with_label=False)
                d = trace_features(p, window_s=WINDOW_S, stride_s=STRIDE_S)
                if d.empty:
                    continue
                d = d[["w_idx"] + [c for c in FEATURE_COLS if c in d.columns]].copy()
                d["family"], d["seed"] = fam, seed
                d["cls"] = "training" if fam in TRAIN_FAMILIES else "serving"
                rows.append(d)
                shutil.rmtree(tmp / "t", ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = pd.concat(rows, ignore_index=True)
    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0
    out.to_parquet(f, index=False)
    return out


# ── evaluation ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

def evaluate(df: pd.DataFrame, shuffle: bool = False, seed: int = 0):
    """Leave-one-family-out, scores POOLED across folds.

    A first version held out one training family and one serving family together
    and took the mean AUC over the 35 pairs. That metric turned out to be nearly
    binary -- a held-out pair is decided as a unit, so its AUC is 0 or 1, sd 0.40
    across pairs -- which is the same all-or-nothing behaviour the power analysis
    found, and it left a standard error of ~0.07 on every condition: far too wide
    to resolve the differences this study is about.

    Pooling instead: each family is held out exactly once, its out-of-fold scores
    are collected, and ONE AUC is computed over all of them. Same leakage
    discipline, far lower variance. Uncertainty comes from bootstrapping over
    FAMILIES, which is the actual unit of replication.

    Returns (pooled_auc, per_family_frame, scores_frame).
    """
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = (df.cls.values == "training").astype(int)
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    fam = df.family.values
    score = np.full(len(y), np.nan)
    per_fam = []
    for f in FAMILIES:
        te = fam == f
        if te.sum() == 0 or len(np.unique(y[~te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(X[~te], y[~te])
        score[te] = clf.predict_proba(X[te])[:, 1]
        per_fam.append(dict(family=f, cls="training" if y[te][0] else "serving",
                            mean_score=float(np.nanmean(score[te])), n=int(te.sum())))
    ok = np.isfinite(score)
    auc = roc_auc_score(y[ok], score[ok]) if len(np.unique(y[ok])) == 2 else np.nan
    return float(auc), pd.DataFrame(per_fam), pd.DataFrame(
        dict(family=fam[ok], y=y[ok], score=score[ok]))


def bootstrap_auc(sc: pd.DataFrame, n: int = 400, seed: int = 0) -> tuple[float, float]:
    """95% CI by resampling FAMILIES, not windows -- windows within a family are
    not independent observations, which is the whole lesson of the power analysis."""
    rng = np.random.default_rng(seed)
    fams = sc.family.unique()
    by = {f: sc[sc.family == f] for f in fams}
    vals = []
    for _ in range(n):
        pick = rng.choice(fams, len(fams), replace=True)
        d = pd.concat([by[f] for f in pick], ignore_index=True)
        if len(np.unique(d.y)) == 2:
            vals.append(roc_auc_score(d.y, d.score))
    if not vals:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(sc_a: pd.DataFrame, sc_b: pd.DataFrame, n: int = 600,
                     seed: int = 0) -> tuple[float, float, float]:
    """CI on (AUC_a - AUC_b), resampling the SAME families for both.

    Unpaired CIs on this population are about +/-0.25 wide, because 12 families
    is few and families differ enormously from each other. But both conditions
    observe the SAME traffic -- only the instrument differs -- so almost all of
    that variance is common and cancels. This is the comparison the study is
    actually about: not "what is the AUC at 1 Hz" but "what does going from
    10 Hz to 1 Hz cost, on these workloads".
    """
    rng = np.random.default_rng(seed)
    fams = np.intersect1d(sc_a.family.unique(), sc_b.family.unique())
    A = {f: sc_a[sc_a.family == f] for f in fams}
    B = {f: sc_b[sc_b.family == f] for f in fams}
    diffs = []
    for _ in range(n):
        pick = rng.choice(fams, len(fams), replace=True)
        da = pd.concat([A[f] for f in pick], ignore_index=True)
        db = pd.concat([B[f] for f in pick], ignore_index=True)
        if len(np.unique(da.y)) == 2 and len(np.unique(db.y)) == 2:
            diffs.append(roc_auc_score(da.y, da.score) - roc_auc_score(db.y, db.score))
    if not diffs:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def main():
    cs = conditions()
    if sys.argv[1] == "features":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        for c in cs[a:b]:
            t = time.time()
            d = build_features(c)
            print(f"  {c.name:16s} {len(d):6d} windows  {time.time()-t:5.1f}s", flush=True)
    elif sys.argv[1] == "eval":
        rows, per_pair, SCORES = [], [], {}
        for c in cs:
            f = CACHE / f"{c.name}.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            auc, pf, sc = evaluate(d)
            lo, hi = bootstrap_auc(sc)
            pf["cond"] = c.name
            per_pair.append(pf); SCORES[c.name] = sc
            rows.append(dict(cond=c.name, knob=c.knob, x=c.x, sample_hz=c.sample_hz,
                             width=c.width, skew_ms=c.skew_ns / 1e6,
                             read_us=c.read_dur_ns / 1000,
                             auc=auc, ci_lo=lo, ci_hi=hi, n_windows=len(d)))
            print(f"  {c.name:16s} AUC {auc:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)
        S = pd.DataFrame(rows); P = pd.concat(per_pair, ignore_index=True)
        # PAIRED comparisons against the ceiling -- the resolvable quantity
        base = SCORES["ceiling"]
        pr = []
        for name, sc in SCORES.items():
            m, lo, hi = paired_bootstrap(sc, base)
            c = next(c for c in cs if c.name == name)
            pr.append(dict(cond=name, knob=c.knob, x=c.x, sample_hz=c.sample_hz,
                           width=c.width, skew_ms=c.skew_ns/1e6,
                           read_us=c.read_dur_ns/1000,
                           d_auc=m, d_lo=lo, d_hi=hi,
                           resolved=bool(hi < 0 or lo > 0)))
        PR = pd.DataFrame(pr); PR.to_csv(OUT / "paired_vs_ceiling.csv", index=False)
        print("\nPAIRED vs ceiling (50 Hz, 64-bit, perfect clocks):")
        for r in PR.sort_values(["knob", "x"]).itertuples():
            mark = "  <-- resolved" if r.resolved else ""
            print(f"  {r.cond:16s} {r.d_auc:+.3f}  [{r.d_lo:+.3f}, {r.d_hi:+.3f}]{mark}")
        # FLOOR: labels shuffled. Must land at 0.5 or something leaks.
        d0 = pd.read_parquet(CACHE / "ceiling.parquet")
        floor = np.mean([evaluate(d0, shuffle=True, seed=s)[0] for s in range(5)])
        # HARDNESS control: how far does VOLUME alone get on this population, at
        # the ceiling? Demonstrates the population is hard instead of asserting
        # it. A first attempt scored 1.000 here and had to be rebuilt.
        VOL = ["total_Bps_ib", "total_Bps_nvlink", "utilisation_ib", "utilisation_nvlink"]
        vol_only = evaluate(d0.assign(**{c: 0.0 for c in FEATURE_COLS if c not in VOL}))[0]
        S.to_csv(OUT / "conditions.csv", index=False)
        P.to_csv(OUT / "per_family.csv", index=False)
        json.dump(dict(floor_shuffled_auc=float(floor),
                       hardness_volume_only_auc=float(vol_only),
                       ceiling_auc=float(S[S.cond == "ceiling"].auc.iloc[0]),
                       n_families=len(FAMILIES), n_seeds=len(SEEDS),
                       duration_s=DURATION, window_s=WINDOW_S),
                  open(OUT / "controls.json", "w"), indent=1)
        print(f"\nFLOOR (shuffled labels): {floor:.3f}   must be ~0.5")
        print(f"HARDNESS (volume features only, at the ceiling): {vol_only:.3f}")
        print(f"CEILING (200 Hz, 64-bit, perfect clocks): "
              f"{S[S.cond=='ceiling'].auc.iloc[0]:.3f}")


if __name__ == "__main__":
    main()
