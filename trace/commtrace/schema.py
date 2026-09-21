"""comm_trace/1 — types, on-disk layout, and validation.

See schema/comm_trace_v1.md for the rationale. The short version:

  * tx and rx are separate columns and are never summed; the schema has no
    field for a total, so the mistake is unrepresentable.
  * the agent records RAW counter values; all unwrapping and unit conversion
    happens in reader.py, where it can be re-done.
  * the ground-truth label lives in manifest.json, never in the observation
    table, so it cannot leak into a feature matrix by accident.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "comm_trace/1"

LINK_TYPES = ("nvlink", "ib", "eth", "pcie")
UNITS = {"bytes": 1, "words4": 4, "flits": 8}      # multiplier to get bytes
WIDTHS = (32, 64)

FLAG_READ_ERROR   = 1 << 0
FLAG_WRAP_SUSPECT = 1 << 1
FLAG_SOURCE_RESET = 1 << 2

OBS_SCHEMA = pa.schema([
    ("t_mono_ns",   pa.int64()),
    ("t_wall_ns",   pa.int64()),
    ("link_key",    pa.string()),
    ("tx_raw",      pa.uint64()),
    ("rx_raw",      pa.uint64()),
    ("read_dur_ns", pa.int32()),
    ("flags",       pa.uint8()),
])


@dataclass
class Link:
    link_key: str
    node_id: str
    dev_id: str
    link_type: str
    source: str
    unit: str = "bytes"
    width_bits: int = 64
    link_id: int = -1
    peak_bw_bytes_per_s: float = 0.0

    def validate(self):
        assert self.link_type in LINK_TYPES, f"bad link_type {self.link_type}"
        assert self.unit in UNITS, f"bad unit {self.unit}"
        assert self.width_bits in WIDTHS, f"bad width {self.width_bits}"
        assert self.peak_bw_bytes_per_s > 0, "peak_bw_bytes_per_s is needed to detect missed wraps"

    @property
    def wrap_seconds(self) -> float:
        """How long until this counter rolls over at line rate. If the sample
        interval exceeds this, deltas are wrong rather than merely missing."""
        return (2 ** self.width_bits) * UNITS[self.unit] / self.peak_bw_bytes_per_s

    def min_safe_hz(self, margin: float = 4.0) -> float:
        return margin / self.wrap_seconds


@dataclass
class Manifest:
    trace_id: str
    sample_hz_nominal: float
    schema_version: str = SCHEMA_VERSION
    agent_version: str = "synthetic/0.1"
    nodes: dict = field(default_factory=dict)      # node_id -> {clock_offset_ns, ...}
    ground_truth: dict | None = None               # OMIT for traces to be scored


def write_trace(out_dir: str | Path, obs: pd.DataFrame, links: list[Link],
                manifest: Manifest) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for l in links:
        l.validate()
    keys = {l.link_key for l in links}
    missing = set(obs.link_key.unique()) - keys
    assert not missing, f"observations reference unknown links: {sorted(missing)[:5]}"
    assert "total" not in obs.columns and "bytes" not in obs.columns, \
        "the schema has no field for summed bytes; keep tx and rx separate"

    tbl = pa.Table.from_pandas(obs[[f.name for f in OBS_SCHEMA]], schema=OBS_SCHEMA,
                               preserve_index=False)
    pq.write_table(tbl, out / "observations.parquet", compression="zstd")
    (out / "links.json").write_text(json.dumps([asdict(l) for l in links], indent=1))
    (out / "manifest.json").write_text(json.dumps(asdict(manifest), indent=1))
    return out


def read_trace(in_dir: str | Path) -> tuple[pd.DataFrame, list[Link], Manifest]:
    d = Path(in_dir)
    obs = pq.read_table(d / "observations.parquet").to_pandas()
    links = [Link(**x) for x in json.loads((d / "links.json").read_text())]
    m = json.loads((d / "manifest.json").read_text())
    assert m["schema_version"] == SCHEMA_VERSION, f"unsupported schema {m['schema_version']}"
    return obs, links, Manifest(**m)
