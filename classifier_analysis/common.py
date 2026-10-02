"""Shared loading, feature sets and models for the classifier analysis.

Unit of analysis: one 30 s window of one two-node job (15 s stride). Every
feature set is scored on exactly the same windows, so differences between rows
of a results table come from the features, never from the sample.

Per-GPU features are Robi Rahman's 174 window statistics (his code), taken on
GPUs 0 and 4 of each node and reduced to the job-window by the median over those
four streams. Network features are network_detector/netfeat.py's.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

NIC_NODE = ["vol", "sym", "dir_abs", "duty", "cv", "acf1", "acf3", "fft_peak_frac",
            "burst_sym", "peak_mean"]
NIC_CROSS = ["xcorr", "xdir", "anti_dir", "pair_sym"]
NIC = NIC_NODE + NIC_CROSS


def gpu_cols() -> list[str]:
    return pd.read_parquet(DATA / "gpu_cols.parquet").col.tolist()


def group_of(col: str) -> str:
    """Physical source of a feature, for grouped importance."""
    if col in ("vol", "duty"):
        return "NIC volume"
    if col in ("sym", "burst_sym", "dir_abs", "pair_sym", "anti_dir"):
        return "NIC balance"
    if col in ("cv", "acf1", "acf3", "fft_peak_frac", "peak_mean", "xcorr", "xdir"):
        return "NIC timing"
    if col.startswith("pcie"):
        return "PCIe"
    if re.match(r"(encoder|decoder)_", col):
        return "Video codec"
    if re.search(r"power|temperature", col):
        return "Power & thermal"
    if "mem" in col:
        return "Memory"
    return "Compute"          # utilisation, SM clock, tensor activity, duty, idle


def feature_sets(gc: list[str]) -> dict[str, list[str]]:
    pcie = [c for c in gc if c.startswith("pcie")]
    nonic = [c for c in gc if not c.startswith("pcie")]
    return {
        "NIC": NIC,
        "NIC without volume": [c for c in NIC if c != "vol"],
        "PCIe": pcie,
        "Interconnect (NIC + PCIe)": NIC + pcie,
        "GPU, no interconnect": nonic,
        "GPU (all 174)": gc,
        "Everything": gc + NIC,
    }


def load(include_adversarial: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """One row per job-window present in both channels, with labels."""
    gc = gpu_cols()
    N = pd.read_parquet(DATA / "net.parquet")
    G = pd.read_parquet(DATA / "gpu.parquet")
    agg = G.groupby(["cell", "rid", "w"])[gc].median().reset_index()
    D = N.merge(agg, on=["cell", "rid", "w"], how="inner")
    if not include_adversarial:
        D = D[D.kind == "honest"]
    D = D.reset_index(drop=True)
    D[NIC_CROSS] = D[NIC_CROSS].fillna(0.0)
    return D, gc


def X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.nan_to_num(df[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def models(seed: int = 42) -> dict:
    """The paper's configurations (classifier/threeway_improved.py), plus a
    linear baseline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=2),
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
            n_jobs=2, verbosity=0),
        "Logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                 max_iter=2000)),
    }


def fit_predict(make, Xtr, ytr, Xte):
    m = make()
    if type(m).__name__ == "XGBClassifier":
        pos = ytr.mean()
        w = np.where(ytr == 1, 0.5 / pos, 0.5 / (1 - pos))
        m.fit(Xtr, ytr, sample_weight=w)
    else:
        m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def splits(D: pd.DataFrame, protocol: str, seed: int):
    """Yields (fold, train_idx, test_idx).

    run       5-fold StratifiedGroupKFold grouped by job -- the paper's headline
              protocol ("5-fold grouped CV", grouped by run).
    workload  the same grouped by cell (repetitions of a workload merged) --
              Robi's stricter workload_group() protocol.
    family    leave one workload family out (Robi's family() definition).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    y = D.is_train.values
    if protocol == "family":
        for i, f in enumerate(sorted(D.family.unique())):
            te = np.where(D.family.values == f)[0]
            tr = np.where(D.family.values != f)[0]
            yield i, tr, te
        return
    groups = D.rid.values if protocol == "run" else D.cell.values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(np.zeros(len(D)), y, groups)):
        yield i, tr, te


def metrics(df: pd.DataFrame, pcol: str = "p", thr: float = 0.5) -> dict:
    """Window- and job-level rates over a table of pooled held-out predictions."""
    from sklearn.metrics import roc_auc_score
    call = df[pcol] >= thr
    y = df.is_train.values
    out = dict(
        TPR=float(call[y == 1].mean()) if (y == 1).any() else np.nan,
        FPR=float(call[y == 0].mean()) if (y == 0).any() else np.nan,
        AUC=float(roc_auc_score(y, df[pcol])) if len(set(y)) == 2 else np.nan,
    )
    out["BalAcc"] = (out["TPR"] + 1 - out["FPR"]) / 2
    j = df.assign(c=call).groupby(["rid", "is_train"]).c.mean().gt(0.5).reset_index()
    out["TPR_job"] = float(j[j.is_train == 1].c.mean())
    out["FPR_job"] = float(j[j.is_train == 0].c.mean())
    return out
