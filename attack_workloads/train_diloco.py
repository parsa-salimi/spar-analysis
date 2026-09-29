#!/usr/bin/env python3
"""ADVERSARIAL training: DiLoCo -- synchronise across nodes only every H steps.

WHAT IT ATTACKS. The network detector (spar-analysis/network_detector) calls a
node's 30 s window "training" when it moves traffic and sends about what it
receives. Ordinary data-parallel training does both every step, because every
step ends in a gradient all-reduce across nodes. DiLoCo (Douillard et al., 2023)
removes the per-step all-reduce: each node trains its own replica for H inner
steps, and only then do the nodes average what they learned. Between syncs the
network is idle. How often it syncs, how much each sync moves, and whether the
sync is a burst or a trickle are the knobs this cell sweeps.

THE ALGORITHM.
  inner   R GPUs per node run ordinary DDP over NVLink: AdamW, every step.
  outer   every H inner steps, each node computes its outer gradient
          delta = anchor - current_params, the nodes average it, and an outer
          Nesterov-SGD step (lr 0.7, momentum 0.9) updates the global model.
Only the outer average crosses the NIC. It is hierarchical: local rank r
exchanges the r-th 1/R of the delta with rank r of every other node, then the
node reassembles the full vector over NVLink -- so each sync puts exactly one
model's worth of bytes per node on the network, not R.

KNOBS THAT CHANGE THE NETWORK SIGNATURE.
  --inner-steps H        sync interval
  --outer-quant          wire format of the outer gradient:
                           none  bf16, 2 bytes/param
                           int8  per-block absmax int8, ~1 byte/param
                           int4  per-block absmax int4, ~0.5 bytes/param
  --schedule burst       exchange at the sync point, then carry on
  --schedule spread      exchange spread evenly over the next H inner steps and
                         applied one round late, with the local replica merged
                         toward the new global model (alpha = --merge-alpha), as
                         in Streaming DiLoCo. The lowest average rate a sync can
                         have: bytes_per_sync / (H * step_time).

LABELLING. Name cells s16_advtrain_diloco_*: "advtrain" routes to ml_training
and keeps the cell out of the honest pool. STEP lines carry kind="train" on
every inner step; the sync step also carries phase=sync.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/train_diloco.py \
        --size 1b --inner-steps 100 --outer-quant none --schedule burst --duration 600

MEMORY. Inner DDP keeps a full replica plus AdamW state on every GPU, and the
anchor and global model add two fp32 copies: ~31 GB for 1b. 7b and up need an
HSDP inner loop and are not supported here.
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads._mn_attack import (Topology, ByteLedger, MODEL_CONFIGS, build_model,
                                      synthetic_batch, flat_params, load_flat,
                                      chunk_bounds, padded)
except ImportError:
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown
    from _mn_attack import (Topology, ByteLedger, MODEL_CONFIGS, build_model,
                            synthetic_batch, flat_params, load_flat, chunk_bounds, padded)


# ── wire formats for the outer gradient ─────────────────────────────────────
# Everything travels as a flat uint8 buffer, so the same all_gather works for
# every format on both NCCL and gloo.

QBLOCK = 4096


def encode(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "fp32":                                   # tests only: exact
        return x.float().contiguous().view(torch.uint8)
    if fmt == "none":
        return x.to(torch.bfloat16).contiguous().view(torch.uint8)
    n = x.numel()
    nb = math.ceil(n / QBLOCK)
    xp = torch.cat([x.float(), x.new_zeros(nb * QBLOCK - n, dtype=torch.float32)]).view(nb, QBLOCK)
    qmax = 127 if fmt == "int8" else 7
    scale = xp.abs().amax(dim=1).clamp_min(1e-12) / qmax
    q = torch.round(xp / scale[:, None]).clamp(-qmax, qmax).to(torch.int16)
    if fmt == "int8":
        body = q.to(torch.int8).contiguous().view(torch.uint8).reshape(-1)
    else:                                               # int4: two per byte
        u = (q + 8).to(torch.uint8).reshape(-1)
        body = (u[0::2] | (u[1::2] << 4)).contiguous()
    return torch.cat([body, scale.float().contiguous().view(torch.uint8)])


def decode(buf: torch.Tensor, fmt: str, n: int) -> torch.Tensor:
    if fmt == "fp32":
        return buf.view(torch.float32)[:n].clone()
    if fmt == "none":
        return buf.view(torch.bfloat16)[:n].float()
    nb = math.ceil(n / QBLOCK)
    scale = buf[-4 * nb:].contiguous().view(torch.float32)
    body = buf[:-4 * nb]
    if fmt == "int8":
        q = body.contiguous().view(torch.int8).float().view(nb, QBLOCK)
    else:
        lo = (body & 0x0F).to(torch.int16) - 8
        hi = (body >> 4).to(torch.int16) - 8
        q = torch.stack([lo, hi], dim=1).reshape(-1).float().view(nb, QBLOCK)
    return (q * scale[:, None]).reshape(-1)[:n]


class Exchange:
    """Inter-node average of this rank's chunk, as one all_gather or as
    `pieces` equal all_gathers issued one per inner step."""

    def __init__(self, topo, ledger, chunk: torch.Tensor, fmt: str, pieces: int, step: int):
        self.topo, self.ledger, self.fmt, self.n = topo, ledger, fmt, chunk.numel()
        buf = encode(chunk, fmt)
        self.pieces = max(1, pieces)
        L = buf.numel()
        self.plen = math.ceil(L / self.pieces)
        self.buf = torch.cat([buf, buf.new_zeros(self.plen * self.pieces - L)]).to(topo.device)
        self.len = L
        self.recv = [torch.empty_like(self.buf) for _ in range(topo.N)]
        self.done = 0
        if self.pieces == 1:
            self.advance(step)

    def advance(self, step: int):
        if self.done >= self.pieces:
            return
        a, b = self.done * self.plen, (self.done + 1) * self.plen
        out = [r[a:b] for r in self.recv]
        piece = self.buf[a:b].contiguous()
        if self.topo.N > 1:
            tmp = [torch.empty_like(piece) for _ in range(self.topo.N)]
            dist.all_gather(tmp, piece, group=self.topo.inter)
            for o, t in zip(out, tmp):
                o.copy_(t)
            self.ledger.add(step, sent=(self.topo.N - 1) * piece.numel(),
                            recv=(self.topo.N - 1) * piece.numel())
        else:
            out[0].copy_(piece)
        self.done += 1

    def complete(self) -> bool:
        return self.done >= self.pieces

    def mean(self) -> torch.Tensor:
        vals = [decode(r[:self.len], self.fmt, self.n) for r in self.recv]
        return torch.stack(vals).mean(dim=0)


def gather_in_node(topo, chunk: torch.Tensor, total_padded: int) -> torch.Tensor:
    if topo.R == 1:
        return chunk.clone()
    parts = [torch.empty_like(chunk) for _ in range(topo.R)]
    dist.all_gather(parts, chunk.contiguous(), group=topo.intra)
    return torch.cat(parts)[:total_padded]


def main():
    ap = argparse.ArgumentParser(description="DiLoCo: rare cross-node synchronisation (evasion)")
    ap.add_argument("--size", default="1b", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--duration", type=int, default=600, help="seconds; 0 = no deadline")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N inner steps (tests)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--inner-steps", type=int, default=100, help="H: inner steps per sync")
    ap.add_argument("--inner-lr", type=float, default=3e-4)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--outer-quant", default="none", choices=["none", "int8", "int4", "fp32"])
    ap.add_argument("--schedule", default="burst", choices=["burst", "spread"])
    ap.add_argument("--merge-alpha", type=float, default=0.5,
                    help="spread schedule: weight kept on the local replica when the "
                         "late global update lands")
    ap.add_argument("--backend", default="auto", choices=["auto", "nccl", "gloo"])
    ap.add_argument("--ranks-per-node", type=int, default=0,
                    help="default LOCAL_WORLD_SIZE; set it to fake nodes on one machine")
    ap.add_argument("--netlog-dir", default="")
    ap.add_argument("--dump-dir", default="", help="tests: save final state per rank")
    args = ap.parse_args()

    topo = Topology(args.backend, args.ranks_per_node)
    dev = topo.device
    H = args.inner_steps
    vocab = MODEL_CONFIGS[args.size][4]
    model = build_model(args.size, dev)
    net = (DDP(model, process_group=topo.intra,
               device_ids=[dev.index] if dev.type == "cuda" else None)
           if topo.R > 1 else model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.inner_lr)
    lossf = nn.CrossEntropyLoss()
    amp = dev.type == "cuda"

    L = sum(p.numel() for p in model.parameters())
    G = padded(flat_params(model), topo.R)            # global model (fp32, full)
    A = G.clone()                                     # anchor: where this round began
    s, e, C = chunk_bounds(L, topo.R, topo.local)
    mom = torch.zeros(C, device=dev)                  # outer momentum, this rank's chunk
    ledger = ByteLedger()
    pending = None
    wire_bytes = {"none": 2, "int8": 1, "int4": 0.5, "fp32": 4}[args.outer_quant]

    def outer_step(avg):
        nonlocal G
        mom.mul_(args.outer_momentum).add_(avg)
        upd = avg + args.outer_momentum * mom         # Nesterov
        G = gather_in_node(topo, G[s:e] - args.outer_lr * upd, G.numel())

    if topo.rank == 0:
        print(f"DiLoCo (ADVERSARIAL training): {args.size}, {topo.N} nodes x {topo.R} ranks, "
              f"H={H}, outer wire={args.outer_quant}, schedule={args.schedule}", flush=True)
        print(f"  {L/1e6:.1f}M params -> ~{L*wire_bytes/2**20:.1f} MiB across the NIC per "
              f"node per sync", flush=True)

    deadline = CollectiveDeadline(args.duration or None, dev)
    step, t_start = 0, time.time()
    while not deadline.expired():
        if args.max_steps and step >= args.max_steps:
            break
        x, y = synthetic_batch(vocab, args.batch_size, args.seq_len, topo.node, topo.local,
                               step, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = net(x)
        loss = lossf(out.reshape(-1, vocab).float(), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if pending is not None:                       # spread: one piece per inner step
            pending.advance(step)
        sync = step % H == 0
        if sync:
            cur = padded(flat_params(model), topo.R)
            if args.schedule == "burst":
                ex = Exchange(topo, ledger, (A - cur)[s:e], args.outer_quant, 1, step)
                outer_step(ex.mean())
                A = G.clone()
                load_flat(model, G[:L])
            else:
                delta = (A - cur)[s:e]
                if pending is not None:               # last round's exchange is complete
                    outer_step(pending.mean())
                    cur = args.merge_alpha * cur + (1 - args.merge_alpha) * G
                    load_flat(model, cur[:L])
                A = cur.clone()
                pending = Exchange(topo, ledger, delta, args.outer_quant, H, step)
        if topo.rank == 0:
            kv = dict(method="diloco", H=H)
            if sync:
                kv["phase"] = "sync"
            emit_step("train", step, tokens=args.batch_size * args.seq_len, passes=1,
                      node=topo.node, rank=topo.rank, **kv)

    el = time.time() - t_start
    extra = dict(steps=step, elapsed_s=round(el, 2), H=H, quant=args.outer_quant,
                 schedule=args.schedule, params=L,
                 predicted_bytes_per_sync_per_node=int(L * wire_bytes))
    ledger.dump(topo, args.netlog_dir or None, extra)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        torch.save(dict(params=flat_params(model).cpu(), G=G[:L].cpu(), steps=step),
                   os.path.join(args.dump_dir, f"rank{topo.rank}.pt"))
    if topo.rank == 0:
        print(f"DiLoCo done: {step} inner steps, {step // H} syncs, {el:.0f}s", flush=True)
    sync_teardown()


if __name__ == "__main__":
    main()
