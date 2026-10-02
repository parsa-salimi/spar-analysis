"""Collapse the per-window prediction pool into the headline tables."""
from __future__ import annotations
import pandas as pd
from . import common as C

ORDER = ["NIC", "NIC without volume", "PCIe", "Interconnect (NIC + PCIe)",
         "GPU, no interconnect", "GPU (all 174)", "Everything"]


def main():
    d = C.RES / "preds"
    frames = [pd.read_parquet(p) for p in d.glob("*.parquet") if p.name != "rule.parquet"]
    P = pd.concat(frames, ignore_index=True)
    rule = pd.read_parquet(d / "rule.parquet")

    tabs = {}
    for proto in ("run", "workload", "family"):
        rows = []
        for (fs, model), g in P[P.protocol == proto].groupby(["fs", "model"]):
            # average metrics over seeds (predictions pooled within a seed)
            per_seed = [C.metrics(s) for _, s in g.groupby("seed")]
            m = pd.DataFrame(per_seed).mean().to_dict()
            m |= dict(fs=fs, model=model)
            rows.append(m)
        rm = C.metrics(rule)
        rm |= dict(fs="— rule (unfitted) —", model="rule")
        rows.append(rm)
        T = pd.DataFrame(rows)
        T["fs"] = pd.Categorical(T.fs, ORDER + ["— rule (unfitted) —"], ordered=True)
        T = T.sort_values(["model", "fs"])
        cols = ["fs", "model", "TPR", "FPR", "BalAcc", "AUC", "TPR_job", "FPR_job"]
        tabs[proto] = T[cols]
        T[cols].to_csv(C.RES / f"ablation_{proto}.csv", index=False)

    pd.set_option("display.width", 220)
    for proto, T in tabs.items():
        print(f"\n================  protocol = {proto}  ================")
        print(T.round(3).to_string(index=False))

    # focused view: best model per feature set, balanced accuracy, each protocol
    print("\n\n######## headline: balanced accuracy, best model per feature set ########")
    best = []
    for proto, T in tabs.items():
        t = T[T.model != "rule"]
        b = t.loc[t.groupby("fs", observed=True).BalAcc.idxmax()]
        for _, r in b.iterrows():
            best.append(dict(protocol=proto, fs=str(r.fs), model=r.model,
                             BalAcc=r.BalAcc, TPR=r.TPR, FPR=r.FPR))
    B = pd.DataFrame(best).pivot(index="fs", columns="protocol", values="BalAcc").reindex(ORDER)
    print(B.round(3).to_string())
    B.to_csv(C.RES / "ablation_headline_balacc.csv")


if __name__ == "__main__":
    main()
