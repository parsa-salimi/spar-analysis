"""Can traffic timing tell gradient synchronisation from mixture-of-experts serving?

Symmetry flags both as training: on Long's 10 Hz Nebius corpus, expert-parallel
serving is as balanced as training. This asks whether the *shape* of the traffic
inside a 30 s window separates them, and whether that needs 10 readings a second.

Population: windows the frozen rule calls training (active and balanced), i.e.
exactly the cases a second stage would have to decide. Features per node, per
window, on tx+rx, averaged over the two nodes:

  cv           std / mean                   how uneven
  peak_mean    p95 / mean                   how spiky
  burst_frac   share of samples above half the window's max
  acf_k        autocorrelation at lag k samples
  fft_peak     share of spectral power in the strongest non-zero frequency
  spec_ent     spectral entropy (0 = one pure period, 1 = flat)
  dom_period_s period of the strongest frequency, seconds

computed on the 10 Hz series and on the same series averaged to 1 Hz.

Test: leave-one-workload-out (each training or serving workload held out in turn),
logistic regression and a small random forest, balanced accuracy over held-out
windows. With one serving workload in this population, holding it out leaves no
negative to learn from, so each EP fold trains on EP windows from the other
runs is not possible either; instead the serving class is held out by run
(3 runs) while training is held out by workload.

    python3 -m classifier_analysis.timing
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import common as C
from network_detector import rule as R

WIN, STRIDE = 30.0, 15.0


def feats(x, hz):
    x = np.asarray(x, float)
    m = x.mean()
    out = {}
    if m <= 0 or len(x) < 8:
        return None
    out["cv"] = x.std() / m
    out["peak_mean"] = np.percentile(x, 95) / m
    out["burst_frac"] = float((x > 0.5 * x.max()).mean())
    for lag_s in (0.1, 0.5, 1.0, 2.0) if hz == 10 else (1.0, 2.0, 3.0):
        k = int(round(lag_s * hz))
        if 0 < k < len(x) - 2:
            a, b = x[:-k], x[k:]
            out[f"acf_{lag_s}s"] = 0.0 if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1])
    p = np.abs(np.fft.rfft((x - m) * np.hanning(len(x)))) ** 2
    p[0] = 0
    if p.sum() > 0:
        q = p / p.sum()
        out["fft_peak"] = float(q.max())
        out["spec_ent"] = float(-(q[q > 0] * np.log(q[q > 0])).sum() / np.log(len(q)))
        f = np.fft.rfftfreq(len(x), 1 / hz)
        out["dom_period_s"] = float(1 / f[q.argmax()]) if f[q.argmax()] > 0 else 0.0
    return out


def windows(H, hz):
    rows = []
    for (cell, rid, it), g in H.groupby(["cell", "rid", "is_train"]):
        per = {}
        for node, d in g.groupby("node"):
            d = d.sort_values("timestamp_epoch")
            tot = (d.ib_tx_mbps.fillna(0) + d.ib_rx_mbps.fillna(0)).values
            t = d.timestamp_epoch.values
            if hz == 1:
                s = np.floor(t).astype(int)
                df = pd.DataFrame({"s": s, "v": tot}).groupby("s").v.mean()
                t, tot = df.index.values.astype(float), df.values
            per[node] = (t, tot)
        t0 = min(v[0][0] for v in per.values()); t1 = max(v[0][-1] for v in per.values())
        w = t0
        while w + WIN <= t1:
            fs = []
            tx_rx = []
            for node, (t, v) in per.items():
                sel = (t >= w) & (t < w + WIN)
                f = feats(v[sel], hz)
                if f:
                    fs.append(f)
            if len(fs) == 2:
                r = {k: float(np.mean([f.get(k, np.nan) for f in fs])) for k in fs[0]}
                r.update(cell=cell, rid=rid, is_train=it, w_abs=w)
                rows.append(r)
            w += STRIDE
    return pd.DataFrame(rows)


def main():
    out = C.RES / "timing"; out.mkdir(parents=True, exist_ok=True)
    H = pd.read_parquet(C.DATA / "nebius_ib_10hz.parquet")
    W = pd.read_parquet(C.DATA / "nebius_windows.parquet")
    _, call = R.call(W)
    pos = W[call.values][["rid", "w_abs"]]
    res = []
    for hz in (10, 1):
        T = windows(H, hz)
        # keep only windows the rule calls training (nearest 1 Hz window start)
        T = T.merge(pos.assign(k=1), on=["rid"], how="inner")
        T = T[(T.w_abs_x - T.w_abs_y).abs() < 1.0].drop(columns=["w_abs_y", "k"]).rename(columns={"w_abs_x": "w_abs"})
        T = T.drop_duplicates(["rid", "w_abs"])
        cols = [c for c in T.columns if c not in ("cell", "rid", "is_train", "w_abs")]
        T[cols] = T[cols].fillna(0.0)
        T.to_parquet(out / f"timing_windows_{hz}hz.parquet", index=False)
        y = T.is_train.values
        for mname, mk in (("Logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))),
                          ("RandomForest", lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=2))):
            P = []
            train_cells = sorted(T[T.is_train == 1].cell.unique())
            ep_runs = sorted(T[T.is_train == 0].rid.unique())
            # each training workload held out (with one EP run held out alongside, rotating)
            for i, c in enumerate(train_cells):
                er = ep_runs[i % len(ep_runs)]
                te = (T.cell == c) | (T.rid == er)
                m = mk().fit(T.loc[~te, cols].values, y[~te.values])
                P.append(T[te][["cell", "rid", "is_train"]].assign(p=m.predict_proba(T.loc[te, cols].values)[:, 1]))
            P = pd.concat(P)
            per = P.assign(c=P.p >= 0.5).groupby(["is_train", "cell"]).c.mean()
            tpr = P[P.is_train == 1].assign(c=lambda d: d.p >= 0.5).groupby("cell").c.mean().mean()
            fpr = P[P.is_train == 0].assign(c=lambda d: d.p >= 0.5).c.mean()
            res.append(dict(hz=hz, model=mname, windows=len(T), train_windows=int(y.sum()), ep_windows=int((1 - y).sum()),
                            TPR_cellavg=round(tpr, 3), FPR=round(fpr, 3), BalAcc=round((tpr + 1 - fpr) / 2, 3)))
            print(f"--- {hz} Hz {mname}\n{per.round(2).to_string()}", flush=True)
        if hz == 10:
            print("\nmedians by workload (10 Hz):")
            print(T.groupby("cell")[cols].median().round(3).to_string())
    R_ = pd.DataFrame(res)
    R_.to_csv(out / "timing_separation.csv", index=False)
    print(R_.to_string(index=False))


if __name__ == "__main__":
    main()
