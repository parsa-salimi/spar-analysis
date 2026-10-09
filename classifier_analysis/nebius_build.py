"""Window tables for Long's Nebius corpus, built with the same feature code as
the GCP corpus so the two can be pooled.

Corpus: data/nebius_benign_phase1 (6 cells) and data/nebius_benign_phase2/20261006
(10 training cells) on Long's PR #3 branch; 10 readings per second, 8 GPUs per node.

For comparability with the GCP corpus (1 Hz), every series is first averaged to
1-second bins; the 10 Hz network series are kept separately for timing features.

  network   netfeat.job_windows on each node's InfiniBand rate (ib_tx/rx_mbps,
            MiB/s, all 8 ports summed by the collector), read from GPU 0's file.
  nvlink    the same node features on NVLink: per node, the sum over its 8 GPUs.
  gpu       Robi's sliding_windows on GPUs 0 and 4 of each node, causal
            run-level features, median per job-window.

    python3 -m classifier_analysis.nebius_build          # resumable; run until "ALL DONE"
Writes classifier_analysis/data/nebius_windows.parquet and nebius_ib_10hz.parquet
"""
from __future__ import annotations
import os, re, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd
from network_detector import netfeat as N
from network_detector.jobs import robi

REPO = Path(os.environ.get("NEBIUS_REPO", Path.home() / "mnt/Bertie/GPU-monitoring"))
BRANCH = os.environ.get("NEBIUS_BRANCH", "pr3")   # williamfowler/GPU-monitoring PR #3 (git fetch origin pull/3/head:pr3)
ROOTS = ("data/nebius_benign_phase1", "data/nebius_benign_phase2")
ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
OUT = Path(__file__).resolve().parent / "data"
CACHE = Path.home() / "scratch/neb_cache"
RE = re.compile(r"(?P<root>nebius_benign_phase\d)(?:/(?P<date>\d{8}))?/(?P<node>node\d)/data/"
                r"interconnect_(?P<tax>W\d+)_(?P<cell>[a-z0-9_]+?)_r(?P<rep>\d+)_.*_gpu(?P<gpu>\d)_[0-9a-f]+\.parquet$")

LABELS = {  # cell -> (is_train, family)
    "dense_ddp": (1, "train_ddp"), "ddp_grad_accum": (1, "train_ddp"),
    "fsdp_2n8": (1, "train_fsdp"), "fsdp_1n8_ibquiet": (1, "train_fsdp_nodelocal"),
    "diloco": (1, "train_diloco"), "diloco_streaming": (1, "train_diloco"),
    "lora_only_ddp": (1, "train_lora"), "powersgd_ddp": (1, "train_ddp_powersgd"),
    "tp_dp_train": (1, "train_tp"), "pp_1f1b_train": (1, "train_pipe"),
    "sft_short": (1, "train_sft"), "moe_ep_train_titan": (1, "train_moe"),
    "disagg_prefill_decode": (0, "infer_disagg"), "local_qwen_infer": (0, "inferrep_qwen"),
    "qwen_tp_decode": (0, "infer_tp"), "olmoe_ep_mixed": (0, "infer_moe_ep"),
}


def git(*a, inp=None):
    return subprocess.run(["git", "-C", str(REPO), *a], input=inp, capture_output=True,
                          env=ENV, check=True).stdout


def files():
    rows = []
    for p in git("ls-tree", "-r", "--name-only", BRANCH, *ROOTS).decode().split():
        m = RE.search(p)
        if m:
            rows.append(dict(path=p, phase=m["root"][-6:], node=int(m["node"][-1]), tax=m["tax"],
                             cell=m["cell"], rep=int(m["rep"]), gpu=int(m["gpu"])))
    return pd.DataFrame(rows)


def read(path):
    ptr = git("show", f"{BRANCH}:{path}").decode()
    oid = re.search(r"oid sha256:([0-9a-f]+)", ptr)[1]
    return pd.read_parquet(REPO / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid)


def to_1hz(d):
    d = d.assign(_s=np.floor(d.timestamp_epoch).astype(np.int64))
    num = d.select_dtypes("number").columns.drop("_s")
    other = [c for c in d.columns if c not in num and c != "_s"]
    a = d.groupby("_s")[list(num)].mean()
    if other:
        a = a.join(d.groupby("_s")[other].first())
    a["timestamp_epoch"] = a.index.astype(float)
    return a.reset_index(drop=True)


def job(cell, rep, g, T):
    frames = {(r.node, r.gpu): read(r.path) for r in g.itertuples()}
    nic, nvl, hz10 = {}, {}, []
    for node in (0, 1):
        d0 = frames[(node, 0)]
        hz10.append(d0[["timestamp_epoch", "ib_tx_mbps", "ib_rx_mbps"]].assign(node=node))
        s0 = to_1hz(d0[["timestamp_epoch", "ib_tx_mbps", "ib_rx_mbps"]].fillna(0))
        nic[node] = pd.DataFrame({"timestamp_epoch": s0.timestamp_epoch,
                                  "nic_tx_mbps": s0.ib_tx_mbps, "nic_rx_mbps": s0.ib_rx_mbps})
        per = [to_1hz(frames[(node, k)][["timestamp_epoch", "nvlink_tx_mbps", "nvlink_rx_mbps"]].fillna(0))
               .set_index("timestamp_epoch") for k in range(8) if (node, k) in frames]
        s = pd.concat(per).groupby(level=0).sum()
        nvl[node] = pd.DataFrame({"timestamp_epoch": s.index.astype(float),
                                  "nic_tx_mbps": s.nvlink_tx_mbps.values, "nic_rx_mbps": s.nvlink_rx_mbps.values})
    nw = N.job_windows(nic).reset_index(drop=True); nw["w"] = np.arange(len(nw))
    vw = N.job_windows(nvl).reset_index(drop=True)
    vw = vw[["w_abs"] + [c for c in vw.columns if c in N.NODE_COLS + ["dir_abs"]]]
    vw = vw.rename(columns={c: f"nvl_{c}" for c in vw.columns if c != "w_abs"})
    gw = []
    for (node, gpu), d in frames.items():
        if gpu not in (0, 4):
            continue
        d = T._normalize_columns(to_1hz(d))
        d["run_id"] = f"{cell}_r{rep}_n{node}g{gpu}"
        d["workload_label"] = cell
        d["threeway_label"] = "ml_training" if LABELS[cell][0] else "ml_inference"
        gw.append(d)
    G = T.sliding_windows(pd.concat(gw, ignore_index=True), 30.0, 15.0, run_level_mode="causal")
    G["w_abs"] = G.window_start
    idx = np.searchsorted(nw.w_abs.values, G.w_abs.values)
    best = []
    for i, t in zip(idx, G.w_abs.values):
        c = [k for k in (i - 1, i) if 0 <= k < len(nw)]
        k = min(c, key=lambda k: abs(nw.w_abs.values[k] - t)) if c else -1
        best.append(k if k >= 0 and abs(nw.w_abs.values[k] - t) <= N.STRIDE / 2 else -1)
    G["w"] = best
    G = G[G.w >= 0]
    gcols = [c for c in T.get_feature_cols(G) if c not in ("w", "w_abs")]
    Gm = G.groupby("w")[gcols].median().reset_index()
    J = nw.merge(Gm, on="w")
    vw = vw.reset_index(drop=True); vw["w"] = np.arange(len(vw))
    J = J.merge(vw.drop(columns="w_abs"), on="w", how="left")
    it, fam = LABELS[cell]
    rid = f"neb_{cell}_r{rep}"
    J = J.assign(cell=f"neb_{cell}", rid=rid, is_train=it, family=fam, kind="honest", corpus="nebius")
    H = pd.concat(hz10, ignore_index=True).assign(cell=f"neb_{cell}", rid=rid, is_train=it, family=fam)
    return J, H


def main(budget_s=150):
    t0 = time.time()
    T, _ = robi()
    CACHE.mkdir(parents=True, exist_ok=True)
    F = files()
    todo = list(F.groupby(["cell", "rep"]))
    for (cell, rep), g in todo:
        pf = CACHE / f"{cell}__r{rep}.pkl"
        if pf.exists():
            continue
        if time.time() - t0 > budget_s:
            print(f"budget reached; {sum((CACHE / f'{c}__r{r}.pkl').exists() for (c, r), _ in todo)}/{len(todo)} jobs done")
            return
        pd.to_pickle(job(cell, rep, g, T), pf)
        print(f"{cell:22s} r{rep} done ({time.time()-t0:.0f}s)", flush=True)
    Js, Hs = zip(*[pd.read_pickle(CACHE / f"{c}__r{r}.pkl") for (c, r), _ in todo])
    W = pd.concat(Js, ignore_index=True); H = pd.concat(Hs, ignore_index=True)
    W.to_parquet(OUT / "nebius_windows.parquet", index=False)
    H.to_parquet(OUT / "nebius_ib_10hz.parquet", index=False)
    print("ALL DONE:", len(W), "windows;", W.rid.nunique(), "jobs;", len(H), "10 Hz rows")


if __name__ == "__main__":
    main()
