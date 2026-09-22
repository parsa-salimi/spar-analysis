"""Put the sibling packages on the path, relative to this file, so the detector
runs from a fresh clone with no environment setup."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "trace", ROOT / "comm_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
