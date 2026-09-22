"""Detector v0 under the A1 symmetry attack.

Question: does the detector survive an adversary who pads inter-node traffic to
look like disaggregated serving, when the detector was trained WITHOUT seeing the
attack?

Protocol
--------
* Population: the artifact study's deliberately hard 7 training / 5 serving
  families (trace/artifacts.py). Four of the five serving families are
  symmetric at node level, so symmetry alone cannot separate most of them --
  something else in the feature set has to.
* Reference instrument: 10 Hz, 64-bit counters, 1 ms NTP skew, 80 us read skew.
* Leave-one-family-out, scores pooled. The classifier is fit on CLEAN windows of
  the other eleven families. The held-out family is scored twice when it is a
  training family: clean, and padded to s* = 0.10 (detector/attack.py). Serving
  families are never padded -- the adversary is the one training.
* AUC_clean pools clean scores for every family. AUC_attacked pools padded scores
  for training families with clean scores for serving families.
* Uncertainty: bootstrap over families, the unit of replication.

Feature sets are defined below with the reason each exists.

Usage:  python3 -m detector.evaluate features   # build/cache, resumable
        python3 -m detector.evaluate eval
"""
from __future__ import annotations
import sys, time, json, shutil, tempfile
from pathlib import Path
import numpy as np, pandas as pd

from . import _paths  # noqa: F401
import artifacts as A
from commtrace.gen import sample_events
from commtrace.reader import FEATURE_COLS
from .features import detector_features, BURST_COLS
from .attack import pad_partition

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent / "results"
CACHE = OUT / "features"
S_TARGET = 0.10
REF = [c for c in A.conditions() if c.name == "hz10_w64"][0]

FEATURE_SETS = {
    # The artifact study's detector. Uses raw symmetry, i.e. the D0 defence.
    "reader (D0)": list(FEATURE_COLS),
    # One-feature detectors: what the attack does to the defence it targets,
    # and to the defence built to resist it.
    "raw symmetry only (D0)": ["symmetry_ib"],
    "burst symmetry only (D1)": ["burst_sym_ib"],
    # Only quantities a constant one-way stream cannot move: burst-based or
    # shift-invariant, plus the unpadded intra-node fabric. Raw symmetry, raw
    # volume, utilisation and cv on the padded fabric are excluded because the
    # adversary sets them; base_asym is excluded because it IS the padding.
    "attack-robust (D1)": ["burst_sym_ib", "burst_Bps_ib",
                           "symmetry_nvlink", "total_Bps_nvlink", "utilisation_nvlink", "cv_nvlink",
                           "acf1_ib", "acf1_nvlink", "fft_peak_frac_ib", "fft_peak_frac_nvlink",
                           "xnode_corr"],
    # Timing alone: can cadence and cross-node synchrony separate training from
    # serving without any directional information at all?
    "timing only": ["acf1_ib", "acf1_nvlink", "fft_peak_frac_ib", "fft_peak_frac_nvlink",
                    "xnode_corr"],
}
SYMMETRIC_SERVING = ["pp_serve_70b", "pp_serve_32b", "moe_serve", "coloc_70b"]


def _one(fam: str, seed: int, version: str) -> pd.DataFrame:
    f = CACHE / f"{fam}__{seed}__{version}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    ev = A.events_for(fam, seed)
    slow = 1.0
    if version == "padded":
        ev = pad_partition(ev, S_TARGET)
        slow = ev.meta["pad_slowdown"]
    tmp = Path(tempfile.mkdtemp())
    try:
        p, _ = sample_events(ev, tmp / "t", coll=REF.collection(seed), with_label=False)
        d = detector_features(p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    d["family"], d["seed"], d["version"], d["pad_slowdown"] = fam, seed, version, slow
    d["cls"] = "training" if fam in A.TRAIN_FAMILIES else "serving"
    d.to_parquet(f, index=False)
    return d


def build(budget_s: float = 150.0) -> bool:
    CACHE.mkdir(parents=True, exist_ok=True)
    t0, todo = time.time(), []
    for fam in A.FAMILIES:
        for seed in A.SEEDS:
            todo.append((fam, seed, "clean"))
            if fam in A.TRAIN_FAMILIES:
                todo.append((fam, seed, "padded"))
    done = 0
    for job in todo:
        if (CACHE / f"{job[0]}__{job[1]}__{job[2]}.parquet").exists():
            done += 1; continue
        if time.time() - t0 > budget_s:
            break
        _one(*job); done += 1
    print(f"{done}/{len(todo)} cached")
    return done == len(todo)


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted(CACHE.glob("*.parquet"))],
                   ignore_index=True)
    for c in FEATURE_COLS + BURST_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def _X(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values


def lofo(df: pd.DataFrame, cols: list[str], families=None, shuffle_seed=None):
    """Returns a frame of per-window scores: family, cls, version, y, score."""
    clean = df[df.version == "clean"]
    fams = families or list(A.FAMILIES)
    clean = clean[clean.family.isin(fams)]
    y_all = (clean.cls.values == "training").astype(int)
    if shuffle_seed is not None:
        y_all = np.random.default_rng(shuffle_seed).permutation(y_all)
    out = []
    for f in fams:
        tr = clean.family.values != f
        if len(np.unique(y_all[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(_X(clean[tr], cols), y_all[tr])
        for version in ("clean", "padded"):
            te = df[(df.family == f) & (df.version == version)]
            if te.empty:
                continue
            y = y_all[~tr][:len(te)] if (shuffle_seed is not None and version == "clean") \
                else np.full(len(te), int(te.cls.iloc[0] == "training"))
            out.append(pd.DataFrame(dict(family=f, cls=te.cls.values, version=version,
                                         y=y, score=clf.predict_proba(_X(te, cols))[:, 1])))
    return pd.concat(out, ignore_index=True)


def pooled(sc: pd.DataFrame, attacked: bool) -> pd.DataFrame:
    if attacked:
        return sc[((sc.cls == "training") & (sc.version == "padded")) |
                  ((sc.cls == "serving") & (sc.version == "clean"))]
    return sc[sc.version == "clean"]


def auc_ci(sc: pd.DataFrame, n: int = 400, seed: int = 0):
    auc = roc_auc_score(sc.y, sc.score)
    fams = sc.family.unique(); rng = np.random.default_rng(seed)
    by = {f: sc[sc.family == f] for f in fams}
    bs = []
    for _ in range(n):
        s = pd.concat([by[f] for f in rng.choice(fams, len(fams), replace=True)])
        if s.y.nunique() == 2:
            bs.append(roc_auc_score(s.y, s.score))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def decisions(df: pd.DataFrame, cols: list[str], serving: list[str], threshold: float = 0.5):
    """Detection rate at a fixed threshold, the metric an attack is judged by.

    AUC is a ranking measure and is the wrong one here: an attacker who drives
    training symmetry to 0.10 against a serving family at 0.00 leaves the
    ranking intact -- AUC stays 1.0 -- while every padded window lands on the
    serving side of a boundary fitted on clean data.

    For each training family held out: fit on clean windows of the other
    training families plus `serving`, then report the fraction of the held-out
    family's windows called training, clean and padded. Serving false-positive
    rate comes from holding out each serving family in turn.
    """
    trf = list(A.TRAIN_FAMILIES); fams = trf + serving
    clean = df[(df.version == "clean") & df.family.isin(fams)]
    rows = []
    for f in fams:
        tr = clean[clean.family != f]
        y = (tr.cls.values == "training").astype(int)
        if len(np.unique(y)) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(_X(tr, cols), y)
        r = dict(family=f, cls="training" if f in trf else "serving")
        for v in ("clean", "padded"):
            te = df[(df.family == f) & (df.version == v)]
            if not te.empty:
                r[f"called_training_{v}"] = float((clf.predict_proba(_X(te, cols))[:, 1] > threshold).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def main_eval():
    df = load()
    rows, per_fam = [], []
    for name, cols in FEATURE_SETS.items():
        sc = lofo(df, cols)
        a_c = auc_ci(pooled(sc, False)); a_a = auc_ci(pooled(sc, True))
        sym = sc[sc.family.isin(list(A.TRAIN_FAMILIES) + SYMMETRIC_SERVING)]
        a_s = auc_ci(pooled(sym, False))
        # Where symmetry is SUPPOSED to matter: training against the one serving
        # family that is directional. Clean vs attacked on this pair is the
        # direct test of whether the attack defeats a defence.
        dis = sc[sc.family.isin(list(A.TRAIN_FAMILIES) + ["disagg_70b"])]
        d_c = roc_auc_score(pooled(dis, False).y, pooled(dis, False).score)
        d_a = roc_auc_score(pooled(dis, True).y, pooled(dis, True).score)
        rows.append(dict(feature_set=name, n_features=len(cols),
                         vs_disagg_clean=d_c, vs_disagg_attacked=d_a,
                         auc_clean=a_c[0], clean_lo=a_c[1], clean_hi=a_c[2],
                         auc_attacked=a_a[0], attacked_lo=a_a[1], attacked_hi=a_a[2],
                         auc_vs_symmetric_serving=a_s[0], sym_lo=a_s[1], sym_hi=a_s[2]))
        pf = sc.groupby(["family", "cls", "version"]).score.mean().unstack("version")
        pf["feature_set"] = name
        per_fam.append(pf.reset_index())
    res = pd.DataFrame(rows)
    floor = auc_ci(pooled(lofo(df, FEATURE_SETS["attack-robust (D1)"], shuffle_seed=0), False))
    slow = (df[df.version == "padded"].groupby("family").pad_slowdown.first())
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "auc.csv", index=False)
    pd.concat(per_fam).to_csv(OUT / "per_family.csv", index=False)
    json.dump(dict(floor_shuffled=floor, s_target=S_TARGET,
                   pad_slowdown=slow.to_dict()), open(OUT / "controls.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"\nfloor (labels shuffled, attack-robust set): {floor[0]:.3f} [{floor[1]:.3f}, {floor[2]:.3f}]")
    print("\nslowdown the attack imposed:", {k: round(v, 2) for k, v in slow.items()})

    # The direct attack-vs-defence test: training against the directional
    # serving family, one-feature detectors, decision at P > 0.5.
    att = []
    for name in ("raw symmetry only (D0)", "burst symmetry only (D1)"):
        d = decisions(df, FEATURE_SETS[name], ["disagg_70b"])
        d["feature_set"] = name
        att.append(d)
    # And the full detectors against the whole population.
    for name in ("reader (D0)", "attack-robust (D1)", "timing only"):
        d = decisions(df, FEATURE_SETS[name], list(A.SERVE_FAMILIES))
        d["feature_set"] = name
        att.append(d)
    att = pd.concat(att, ignore_index=True)
    att.to_csv(OUT / "decisions.csv", index=False)
    summ = att.groupby(["feature_set", "cls"])[["called_training_clean", "called_training_padded"]].mean()
    print("\nfraction of windows called training (P > 0.5), mean over held-out families:")
    print(summ.round(3).to_string())
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "features":
        build()
    else:
        main_eval()
