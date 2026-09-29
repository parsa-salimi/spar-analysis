"""The job table: one row per multi-node job, with its label, family and flags.

Labels and families are Robi Rahman's, imported from his branch, so every
number here is comparable to his multi-node analysis:
  label   threeway_label(cell)  -> ml_training / ml_inference / other
  family  family(cell) from multinode_family_holdout.py: scope prefix and
          parameter suffixes removed, sweep variants merged.

Two departures, both stated:
  * `ddpstagger` and `ddpkaccum` are evasion strategies from adversarial_ddp.py;
    their names lack "advtrain", so his honest mask keeps them. Here they are
    adversarial.
  * Invalid jobs are dropped: files moved to quarantine, and the jobs listed in
    gcs_runs_manifest/invalidated_markers.txt where the reason is that nothing
    ran (s16_infershard_vllm_tp8pp2: node0 failed, node1 recorded idle GPUs).
    s16_train_fsdp7b is kept -- it was invalidated only for a batch-size
    confound in the scope ladder, and it is genuine training.
"""
from __future__ import annotations
import re, sys, subprocess, tarfile, io
import pandas as pd
from . import data as D

ROBI = D.CACHE / "robi_code"
ADV = re.compile(r"(advtrain|adversarial|whitebox|symgemm|typeconf|stalled|ddpstagger|ddpkaccum)", re.I)
INVALID_CELLS = {"s16_infershard_vllm_tp8pp2"}


def robi():
    """Robi's classifier package, extracted from the branch (read-only)."""
    if not (ROBI / "classifier" / "threeway_improved.py").exists():
        ROBI.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "-C", str(D.REPO), "archive", D.BRANCH, "classifier"],
                             capture_output=True, env=D.ENV, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(ROBI)
    if str(ROBI) not in sys.path:
        sys.path.insert(0, str(ROBI))
    import classifier.threeway_improved as T
    import classifier.rebuttal_analysis.multinode_family_holdout as F
    return T, F


def table() -> pd.DataFrame:
    T, F = robi()
    ix = D.index()
    ix = ix[~ix.quarantined & ~ix.cell.isin(INVALID_CELLS)]
    ix = ix[~ix.cell.str.startswith("smoke_")]
    g = ix.groupby(["cell", "rid"])
    jobs = g.agg(session=("session", "first"),
                 n_gpu_files=("kind", lambda k: int((k == "gpu").sum())),
                 n_nic_files=("kind", lambda k: int((k == "nic").sum())),
                 gpu_avail=("available", lambda a: 0)).reset_index()
    avail = ix.groupby(["cell", "rid", "kind"]).available.sum().unstack(fill_value=0)
    jobs = jobs.merge(avail.rename(columns={"gpu": "gpu_files_avail", "nic": "nic_files_avail"})
                      .reset_index(), on=["cell", "rid"], how="left").drop(columns="gpu_avail")
    jobs["scope"] = jobs.cell.str.extract(r"^(s16|s8|s1)_")[0]
    jobs["label"] = jobs.cell.map(T.threeway_label)
    jobs["is_train"] = (jobs.label == "ml_training").astype(int)
    jobs["family"] = jobs.cell.map(F.family)
    jobs["kind"] = jobs.cell.str.contains(ADV).map({True: "adversarial", False: "honest"})
    return jobs


if __name__ == "__main__":
    j = table()
    s = j[j.scope == "s16"]
    print(f"s16 jobs {len(s)}  cells {s.cell.nunique()}  families {s.family.nunique()}")
    print(s.groupby(["kind", "label"]).size().to_string())
    print("\nfamilies:", sorted(s[s.kind == 'honest'].family.unique()))
