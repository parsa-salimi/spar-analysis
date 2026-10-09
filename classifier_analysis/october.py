"""The October corpus (Robi's main branch, sessions to 2026-10-03): the
pre-registered test of the frozen rule, per-workload rule behaviour, and the
first measured DiLoCo-family runs against the window-coverage model.

  prereg.csv            each row of network_detector/PREDICTIONS.md, scored
  cell_rule.csv         every two-node workload: activity, symmetry, rule call
  sync_coverage.csv     LocalSGD / DiLoCo runs: measured sync period and burst
                        length from the 1 Hz NIC series, predicted share of
                        windows that contain a sync, measured share

    python3 -m classifier_analysis.october
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import common as C
from network_detector import rule as R

OUT = C.RES / "october"
WIN, STRIDE = 30.0, 15.0

PRED = [  # (cells regex, class, predicted call) -- verbatim from PREDICTIONS.md
    (r"s16_train_lora", "training", "training"),
    (r"s16_train_contrastive$", "training", "training"),
    (r"s16_train_diffusion$", "training", "training"),
    (r"s16_train_pipe7b$", "training", "training"),
    (r"s16_advtrain_pipe(1b|70b)", "training", "training"),
    (r"s16_inferrec_dlrm", "inference", "false alarm"),
    (r"s16_nonml_fftshard", "other", "false alarm"),
    (r"s16_nonml_stencil$", "other", "false alarm"),
    (r"s16_nonml_nccl_bw$", "other", "false alarm"),
    (r"s16_(infervis_vitl|inferembed_|inferdiff_sdxl|inferasr_whisper|infercnn_rn50|inferhf_qwen7b)", "inference", "no call"),
    (r"s16_inferhf_shard72b", "inference", "no call"),
    (r"s16_(infermem_|inferserve_)", "inference", "cleared"),
]


def per_job(N):
    a, c = R.call(N)
    N = N.assign(active=a.values, call=c.values, rate=10 ** N.vol - 1)
    return N.groupby(["kind", "label", "family", "cell", "rid"]).agg(
        windows=("w", "size"), active=("active", "mean"), called=("call", "mean"),
        sym=("sym", "median"), rate_mibps=("rate", "median")).reset_index()


def prereg(J):
    rows = []
    for rx, cls, pred in PRED:
        g = J[J.cell.str.contains(rx.replace("(", "(?:"))]
        maj = g.called > 0.5
        act = g.active > 0.5
        if pred in ("training", "false alarm"):
            ok = maj
        elif pred == "no call":
            ok = ~act & ~maj
        else:  # cleared: active but not called
            ok = ~maj
        rows.append(dict(cells=rx.replace("s16_", ""), cls=cls, predicted=pred, jobs=len(g),
                         jobs_matching=int(ok.sum()), windows_called=round(g.called.mean(), 3),
                         median_sym=round(g.sym.median(), 3),
                         median_rate_mibps=round(g.rate_mibps.median(), 3)))
    return pd.DataFrame(rows)


def sync_coverage(S, J):
    rows = []
    cells = sorted(c for c in S.cell.unique() if any(k in c for k in ("localsgd", "diloco")))
    for c in cells:
        for rid, g in S[S.cell == c].groupby("rid"):
            per = []
            for _, d in g.groupby("node"):
                d = d.sort_values("timestamp_epoch")
                on = (d.nic_tx_mbps + d.nic_rx_mbps).values >= R.ACTIVE_MIBPS
                starts = np.flatnonzero(np.diff(on.astype(int)) == 1)
                runs = [len(x) for x in np.split(on, np.flatnonzero(np.diff(on.astype(int))) + 1) if x.size and x[0]]
                per.append((len(d), on.mean(), len(starts),
                            np.median(np.diff(starts)) if len(starts) > 1 else np.nan,
                            np.median(runs) if runs else 0.0))
            n, on_frac, nb, T, b = per[0]
            pred = min(1.0, (WIN + b) / T) if T == T and T > 0 else np.nan
            j = J[(J.cell == c) & (J.rid == rid)]
            rows.append(dict(cell=c.replace("s16_train_", ""), rid=rid[:6], seconds=n, bursts=nb,
                             period_s=T, burst_s=b, wire_busy=round(on_frac, 3),
                             predicted_windows_with_sync=round(pred, 3) if pred == pred else np.nan,
                             measured_windows_active=round(float(j.active.iloc[0]), 3) if len(j) else np.nan,
                             measured_windows_called=round(float(j.called.iloc[0]), 3) if len(j) else np.nan))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    N = pd.read_parquet(C.DATA / "net.parquet")
    S = pd.read_parquet(C.DATA / "nic_1hz.parquet")
    J = per_job(N)
    cell = J.groupby(["kind", "label", "family", "cell"]).agg(
        jobs=("rid", "size"), windows=("windows", "sum"), active=("active", "mean"),
        called=("called", "mean"), sym=("sym", "median"), rate_mibps=("rate_mibps", "median"),
        majority=("called", lambda x: float((x > 0.5).mean())),
        any_window=("called", lambda x: float((x > 0).mean()))).reset_index()
    cell.round(3).to_csv(OUT / "cell_rule.csv", index=False)
    P = prereg(J); P.to_csv(OUT / "prereg.csv", index=False)
    Q = sync_coverage(S, J); Q.to_csv(OUT / "sync_coverage.csv", index=False)
    pd.set_option("display.width", 250)
    print(P.to_string(index=False)); print(); print(Q.to_string(index=False))


if __name__ == "__main__":
    main()
