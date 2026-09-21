"""Paths, and the one external dependency this analysis has.

The feature extractor and label taxonomy live in the team repo
(github.com/williamfowler/GPU-monitoring), which is private and carries ~2.8 GB
of Git-LFS telemetry, so it is NOT vendored here. Point `GPU_MONITORING_ROOT` at
your clone, or keep the clone next to this repo and it is found automatically.

Note for anyone cloning that repo: GitHub's web-UI "Download ZIP" does not fetch
LFS objects -- you get 130-byte pointer files. Run `git lfs install` and clone.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA    = HERE / "data"
FIGURES = HERE / "figures"
for _d in (RESULTS, DATA, FIGURES):
    _d.mkdir(exist_ok=True)

_MARKER = Path("classifier") / "threeway_improved.py"

def gpu_monitoring_root() -> Path:
    cands = []
    if os.environ.get("GPU_MONITORING_ROOT"):
        cands.append(Path(os.environ["GPU_MONITORING_ROOT"]).expanduser())
    cands += [HERE.parent.parent / "GPU-monitoring",      # sibling of this repo
              Path.home() / "mnt" / "GPU-monitoring",     # Cowork mount
              Path("D:/Bertie/GPU-monitoring")]
    for c in cands:
        if (c / _MARKER).exists():
            return c.resolve()
    raise SystemExit(
        "Could not find the GPU-monitoring checkout.\n"
        "Set GPU_MONITORING_ROOT, or clone it next to this repo:\n"
        "  git lfs install\n"
        "  git clone https://github.com/williamfowler/GPU-monitoring.git\n"
        f"Looked in: {', '.join(str(c) for c in cands)}")

GPU_ROOT = gpu_monitoring_root()
if str(GPU_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
