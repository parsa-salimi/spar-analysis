"""Shared plumbing for the network-aimed attack workloads.

Used by train_diloco.py and train_ps_lopsided.py. Belongs next to them in
workloads/, alongside _step.py and dist_deadline.py.

Topology. torchrun numbers ranks contiguously per node, so with R ranks per node
rank = node * R + local. Two families of process groups follow:

  intra[node]   the R ranks of one node          traffic stays on NVLink
  inter[local]  rank `local` of every node        traffic crosses the NIC

Every cross-node byte these workloads move goes through `inter` or through an
explicit point-to-point send between nodes, and is recorded in a ByteLedger so a
run can report exactly what it put on the network, step by step.

CPU testing. With `--backend gloo --ranks-per-node R` the same code runs as W
processes on one machine pretending to be W/R nodes. The `tiny` model config
exists only for that.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.distributed as dist

try:
    from workloads.fsdp_training import MODEL_CONFIGS, RandomTransformerLM
except ImportError:
    from fsdp_training import MODEL_CONFIGS, RandomTransformerLM

# Test-only configuration: (hidden, layers, heads, ffn, vocab).
MODEL_CONFIGS.setdefault("tiny", (32, 2, 4, 64, 128))


class Topology:
    def __init__(self, backend: str = "auto", ranks_per_node: int = 0):
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        self.backend = backend
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.R = ranks_per_node or int(os.environ.get("LOCAL_WORLD_SIZE", self.world))
        if self.world % self.R:
            raise SystemExit(f"world {self.world} is not a multiple of ranks-per-node {self.R}")
        self.N = self.world // self.R
        self.node, self.local = divmod(self.rank, self.R)
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", self.local)))
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cpu")
        # Every rank must create every group, in the same order.
        self.intra = None
        for n in range(self.N):
            g = dist.new_group([n * self.R + r for r in range(self.R)])
            if n == self.node:
                self.intra = g
        self.inter = None
        for r in range(self.R):
            g = dist.new_group([n * self.R + r for n in range(self.N)])
            if r == self.local:
                self.inter = g

    def peer(self, node: int) -> int:
        """Global rank of this rank's counterpart on `node`."""
        return node * self.R + self.local


class ByteLedger:
    """Bytes this rank puts on / takes off the inter-node network, per step.

    Counted from payload sizes at the call site, with the standard ring cost for
    collectives: an all_gather over N members sends and receives (N-1) chunks.
    The kernel's NIC counters will read slightly more (headers, the deadline
    vote); the point is the shape -- when bytes move, and in which direction.
    """

    def __init__(self):
        self.steps: dict[int, list[int]] = {}

    def add(self, step: int, sent: int = 0, recv: int = 0):
        s = self.steps.setdefault(step, [0, 0])
        s[0] += int(sent)
        s[1] += int(recv)

    def summary(self) -> dict:
        tx = sum(v[0] for v in self.steps.values())
        rx = sum(v[1] for v in self.steps.values())
        return dict(tx_bytes=tx, rx_bytes=rx,
                    symmetry=(min(tx, rx) / max(tx, rx)) if max(tx, rx) else 1.0,
                    steps_with_traffic=sum(1 for v in self.steps.values() if v[0] or v[1]))

    def dump(self, topo: Topology, path: str | None, extra: dict | None = None):
        rec = dict(rank=topo.rank, node=topo.node, local=topo.local, **self.summary(),
                   per_step={k: v for k, v in sorted(self.steps.items())}, **(extra or {}))
        print("NETLOG " + json.dumps({k: rec[k] for k in rec if k != "per_step"}), flush=True)
        if path:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"netlog_rank{topo.rank}.json"), "w") as f:
                json.dump(rec, f)


def build_model(size: str, device) -> torch.nn.Module:
    torch.manual_seed(0)                   # identical initial weights on every rank
    return RandomTransformerLM(size).to(device)


def synthetic_batch(vocab: int, batch: int, seq: int, node: int, local: int, step: int,
                    device):
    """Random tokens, deterministic in (node, local rank, step): different data
    on every rank, reproducible by a reference implementation."""
    g = torch.Generator().manual_seed(1_000_003 * node + 10_007 * local + step)
    x = torch.randint(0, vocab, (batch, seq), generator=g)
    y = torch.randint(0, vocab, (batch, seq), generator=g)
    return x.to(device), y.to(device)


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).float() for p in model.parameters()])


def load_flat(model, flat: torch.Tensor):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p).to(p.dtype))
            i += n


def chunk_bounds(total: int, parts: int, idx: int) -> tuple[int, int, int]:
    """Equal chunks of a flat vector padded to a multiple of `parts`.
    Returns (start, end, chunk_len); `end` may exceed `total` (padding)."""
    c = math.ceil(total / parts)
    return idx * c, (idx + 1) * c, c


def padded(flat: torch.Tensor, parts: int) -> torch.Tensor:
    c = math.ceil(flat.numel() / parts)
    if c * parts == flat.numel():
        return flat
    return torch.cat([flat, flat.new_zeros(c * parts - flat.numel())])


def now() -> float:
    return time.time()
