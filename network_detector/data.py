"""Read Robi Rahman's multi-node corpus straight from its git branch.

The corpus lives on the `multinode-tpu-experiments` branch of GPU-monitoring,
which is not checked out, and most of its parquet files are Git LFS pointers.
This module reads every file without touching the clone's working tree:

  * a small file stored in git itself is read with `git cat-file`;
  * an LFS pointer is resolved to `.git/lfs/objects/<aa>/<bb>/<oid>`, which is
    present once `git lfs fetch origin multinode-tpu-experiments` has been run.

Files are extracted once into CACHE (outside the repo) and indexed.

File names carry the metadata (scripts/rebuttal_collection/multinode_queue.sh):
  per-GPU   {cell}_node{r}_gpu{i}_{GPU_NAME}_{rid12}.parquet
  NIC       nic_{cell}_node{r}_{rid12}[_truncated].parquet[.partial]
`rid12` is sha1(RUN_STAMP_cell_rep)[:12], shared by every file of one job.
"""
from __future__ import annotations
import os, re, subprocess, hashlib
from pathlib import Path
import pandas as pd

REPO = Path(os.environ.get("GPUMON_REPO",
            Path.home() / "mnt/Bertie/GPU-monitoring-new"))
BRANCH = os.environ.get("GPUMON_BRANCH", "origin/multinode-tpu-experiments")
CACHE = Path(os.environ.get("MN_CACHE", Path.home() / "scratch/mn_cache"))
ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

LFS_RE = re.compile(rb"^version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)")
NIC_RE = re.compile(r"^(?:.*__)?nic_(?P<cell>.+)_node(?P<node>\d+)_(?P<rid>[0-9a-f]{12})"
                    r"(?P<trunc>_truncated)?\.parquet(?P<partial>\.partial)?$")
GPU_RE = re.compile(r"^(?P<cell>(?:smoke_)?s(?:1|8|16)_.+?)_node(?P<node>\d+)_gpu(?P<gpu>\d+)_"
                    r"(?P<gpuname>.+)_(?P<rid>[0-9a-f]{12})\.parquet$")


def _git(*args, inp=None) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], input=inp,
                          capture_output=True, env=ENV, check=True).stdout


def list_files(prefixes=("data/gcp", "gcs_c_quarantine")) -> pd.DataFrame:
    out = _git("ls-tree", "-r", BRANCH, "--", *prefixes).decode()
    rows = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        if typ == "blob":
            rows.append((path, sha))
    return pd.DataFrame(rows, columns=["path", "sha"])


def _batch_read(shas: list[str]) -> dict[str, bytes]:
    """git cat-file --batch for many small blobs in one process."""
    if not shas:
        return {}
    raw = _git("cat-file", "--batch", inp=("\n".join(shas) + "\n").encode())
    res, i = {}, 0
    for sha in shas:
        nl = raw.index(b"\n", i)
        hdr = raw[i:nl].split()
        size = int(hdr[2])
        res[sha] = raw[nl + 1: nl + 1 + size]
        i = nl + 1 + size + 1
    return res


def index(refresh: bool = False) -> pd.DataFrame:
    """One row per multi-node file: kind (nic|gpu), parsed name fields, where
    its bytes are, and whether they are available locally."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ip = CACHE / "index.parquet"
    if ip.exists() and not refresh:
        return pd.read_parquet(ip)
    f = list_files()
    f["name"] = f.path.map(os.path.basename)
    f["session"] = f.path.map(lambda p: p.split("/")[2] if p.startswith("data/gcp/") else "quarantine")
    rows = []
    for r in f.itertuples():
        m = NIC_RE.match(r.name)
        if m:
            rows.append(dict(path=r.path, sha=r.sha, session=r.session, kind="nic",
                             cell=m["cell"], node=int(m["node"]), gpu=-1, rid=m["rid"],
                             partial=bool(m["partial"]), truncated=bool(m["trunc"])))
            continue
        m = GPU_RE.match(r.name)
        if m:
            rows.append(dict(path=r.path, sha=r.sha, session=r.session, kind="gpu",
                             cell=m["cell"], node=int(m["node"]), gpu=int(m["gpu"]),
                             rid=m["rid"], partial=False, truncated=False,
                             gpuname=m["gpuname"]))
    ix = pd.DataFrame(rows)
    # small blobs: read their heads to tell real parquet from LFS pointer
    blobs = _batch_read(ix.sha.tolist())
    oid, size, lfs = [], [], []
    for sha in ix.sha:
        b = blobs[sha]
        m = LFS_RE.match(b)
        lfs.append(bool(m)); oid.append(m[1].decode() if m else "")
        size.append(int(m[2]) if m else len(b))
    ix["lfs"], ix["oid"], ix["bytes"] = lfs, oid, size
    obj = REPO / ".git" / "lfs" / "objects"
    ix["available"] = [(not l) or (obj / o[:2] / o[2:4] / o).exists()
                       for l, o in zip(ix.lfs, ix.oid)]
    ix["quarantined"] = ix.session.isin(["quarantine", "_quarantine_idle"])
    ix.to_parquet(ip, index=False)
    return ix


def read(row) -> pd.DataFrame:
    """Load one indexed file as a DataFrame, extracting to CACHE on first use."""
    dst = CACHE / "files" / f"{row.sha}.parquet"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if row.lfs:
            src = REPO / ".git/lfs/objects" / row.oid[:2] / row.oid[2:4] / row.oid
            dst.write_bytes(src.read_bytes())
        else:
            dst.write_bytes(_batch_read([row.sha])[row.sha])
    return pd.read_parquet(dst)


if __name__ == "__main__":
    ix = index(refresh=True)
    print(ix.groupby(["kind", "available"]).size().to_string())
    print("\nsessions:", ix.groupby("session").size().to_dict())
    n = ix[ix.kind == "nic"]
    print(f"\nNIC files {len(n)}, distinct jobs {n.rid.nunique()}, cells {n.cell.nunique()}")
