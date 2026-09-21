"""Path resolution for the trace package (mirrors power_analysis/config.py)."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO / "comm_model")):
    if p not in sys.path:
        sys.path.insert(0, p)
