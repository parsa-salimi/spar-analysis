"""The FROZEN two-stage training detector.

Frozen 2026-10-12. This file plus the two serialised models in
`frozen/` define the detector completely. Do not edit after the freeze: a
change here voids the pre-registration in FROZEN_DETECTOR.md. To change the
detector, make a new dated freeze.

Design (unchanged from classifier_analysis/combined.py, "stage 2 OR GPU-quiet"):

  stage 1   the frozen network rule (network_detector/rule.py):
            active  = mean(tx+rx) >= 1 MiB/s per node
            fires   = active and symmetry >= 0.8
  stage 2   a RandomForest on the 174 GPU features, fitted ONLY on windows
            the rule fires on; decides whether that balanced traffic is training.
  fallback  a RandomForest on the same 174 GPU features, fitted on ALL windows;
            used where the rule is silent, to catch training with no balanced
            cross-node trace.
  verdict   training if (rule fires AND stage 2 says training)
                     or (rule silent AND fallback says training).

Both forests: n_estimators=200, min_samples_leaf=2, max_features="sqrt",
class_weight="balanced", random_state=0. Cutoff 0.5.

Training corpus: every honest window of Robi's GCP corpus as exported into
classifier_analysis/data/ (net.parquet + gpu.parquet + jobs.parquet). Adversarial
jobs are never trained on. The models are fit once, here, and serialised; scoring
never refits.

    python3 -m classifier_analysis.detector_frozen freeze    # train + serialise + manifest
    python3 -m classifier_analysis.detector_frozen verify    # reload + reproduce manifest hashes
"""
from __future__ import annotations
import hashlib, json, sys, datetime, platform
from pathlib import Path
import numpy as np
import pandas as pd
from . import common as C
from network_detector import rule as R

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen"
FREEZE_DATE = "2026-10-12"
CUTOFF = 0.5


def rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=0, n_jobs=2)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_training():
    """Honest GCP windows, the exact table every experiment used."""
    D, gc = C.load()                      # honest only, GCP
    return D.reset_index(drop=True), gc


def freeze():
    import joblib, sklearn
    FROZEN.mkdir(exist_ok=True)
    D, gc = _load_training()
    y = D.is_train.values
    Xg = C.X(D, gc)
    _, call = R.call(D)
    pos = call.values.astype(bool)

    fallback = rf().fit(Xg, y)                       # all windows
    stage2 = rf().fit(Xg[pos], y[pos])               # rule-positive windows only

    joblib.dump(fallback, FROZEN / "fallback_gpu.joblib")
    joblib.dump(stage2, FROZEN / "stage2_gpu.joblib")
    pd.Series(gc, name="col").to_frame().to_parquet(FROZEN / "gpu_cols.parquet", index=False)

    manifest = {
        "name": "two-stage training detector",
        "frozen": FREEZE_DATE,
        "design": "stage2 OR GPU-quiet (classifier_analysis/combined.py)",
        "rule": {"active_mibps": R.ACTIVE_MIBPS, "balanced": R.BALANCED,
                 "rule_frozen": "2026-09-24", "source": "network_detector/rule.py"},
        "cutoff": CUTOFF,
        "forest": {"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt",
                   "class_weight": "balanced", "random_state": 0},
        "features": {"rule": ["vol", "sym"], "stage2": gc, "fallback": gc,
                     "n_gpu_features": len(gc)},
        "training_corpus": {
            "source": "Robi GCP corpus, honest jobs only, as exported to classifier_analysis/data/",
            "windows": int(len(D)), "training_windows": int(y.sum()),
            "rule_positive_windows": int(pos.sum()),
            "stage2_train_windows": int(pos.sum()),
            "stage2_train_positive": int(y[pos].sum()),
            "jobs": int(D.rid.nunique()), "families": int(D.family.nunique())},
        "files": {
            "module": _sha(HERE / "detector_frozen.py"),
            "rule": _sha(HERE.parent / "network_detector" / "rule.py"),
            "stage2_gpu.joblib": _sha(FROZEN / "stage2_gpu.joblib"),
            "fallback_gpu.joblib": _sha(FROZEN / "fallback_gpu.joblib"),
            "gpu_cols.parquet": _sha(FROZEN / "gpu_cols.parquet")},
        "environment": {"python": platform.python_version(),
                        "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "built": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (FROZEN / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def load():
    """Return (stage2, fallback, gpu_cols) from the frozen files."""
    import joblib
    gc = pd.read_parquet(FROZEN / "gpu_cols.parquet").col.tolist()
    return (joblib.load(FROZEN / "stage2_gpu.joblib"),
            joblib.load(FROZEN / "fallback_gpu.joblib"), gc)


def score(D: pd.DataFrame) -> pd.DataFrame:
    """Score a window table (net features + GPU features merged, as common.load
    produces). Returns the per-window verdict and the two probabilities."""
    stage2, fallback, gc = load()
    Xg = C.X(D, gc)
    active, call = R.call(D)
    rule_fires = call.values.astype(bool)
    p_fallback = fallback.predict_proba(Xg)[:, 1]
    p_stage2 = np.full(len(D), np.nan)
    if rule_fires.any():
        p_stage2[rule_fires] = stage2.predict_proba(Xg[rule_fires])[:, 1]
    verdict = np.where(rule_fires, p_stage2 >= CUTOFF, p_fallback >= CUTOFF)
    out = D[["cell", "rid"]].copy() if "cell" in D else pd.DataFrame(index=D.index)
    out["rule_fires"] = rule_fires
    out["p_stage2"] = p_stage2
    out["p_fallback"] = p_fallback
    out["training"] = verdict.astype(bool)
    return out


def job_verdict(scored: pd.DataFrame, key="rid") -> pd.DataFrame:
    """Per-job calls: majority of windows, and any window."""
    g = scored.groupby(key).training
    return pd.DataFrame({"majority": g.mean() > 0.5, "any_window": g.max() > 0}).reset_index()


def verify():
    """Reload the frozen models and confirm the manifest hashes still match."""
    m = json.loads((FROZEN / "MANIFEST.json").read_text())
    ok = True
    for f, h in m["files"].items():
        p = (HERE / "detector_frozen.py") if f == "module" else \
            (HERE.parent / "network_detector" / "rule.py") if f == "rule" else (FROZEN / f)
        cur = _sha(p)
        status = "ok" if cur == h else "CHANGED"
        if cur != h:
            ok = False
        print(f"{status:8s} {f}: manifest {h} current {cur}")
    print("FROZEN INTACT" if ok else "WARNING: files changed since freeze")
    return ok


if __name__ == "__main__":
    {"freeze": freeze, "verify": verify}.get(sys.argv[1] if len(sys.argv) > 1 else "freeze", freeze)()
