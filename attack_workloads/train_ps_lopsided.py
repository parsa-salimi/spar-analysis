#!/usr/bin/env python3
"""ADVERSARIAL training: a lopsided parameter server.

WHAT IT ATTACKS. The network detector (spar-analysis/network_detector) relies on
one property of data-parallel training: the gradient all-reduce sends as much as
it receives on every node, so training reads "balanced" (symmetry ~0.96 on
Robi's NIC), while pipeline-sharded serving reads one-directional (0.04-0.18).
That is a property of the ALL-REDUCE, not of training. This cell trains with a
parameter server instead, and compresses only the upload, so every node's
traffic is lopsided -- the signature of a pipeline-serving stage.

THE ALGORITHM, every step.
  1. Every rank computes gradients on its own batch; each node averages them
     over NVLink (intra-node all_reduce).
  2. Node 0 is the server. It also trains. Each worker node compresses its
     gradient: top-k by magnitude keeping a fraction c of the entries, with
     error feedback (entries not sent accumulate and go next time). It uploads
     values + int32 indices to node 0.
  3. Node 0 adds the workers' sparse gradients to its own, averages, and takes
     an AdamW step on the fp32 master weights.
  4. Node 0 sends the updated weights back DENSE, in bf16.
As in train_diloco.py, rank `local` on each node handles only the local-th 1/R
of the model, so each node moves one model's worth, not R.

EXPECTED SIGNATURE (two nodes, bf16 wire), per step:
  server node  sends 2 bytes/param, receives 6c bytes/param (2 value + 4 index)
  worker node  the mirror image
  => per-node symmetry ~= 3c:  c = 0.05 -> 0.15, c = 0.1 -> 0.3, c = 0.3 -> 0.9
At two nodes this costs nothing in bandwidth -- the server's link carries the
same bytes as the ring all-reduce it replaces. With more nodes the server's
link carries (N-1) models per step and becomes the bottleneck.

LABELLING. Name cells s16_advtrain_pslop_*: "advtrain" routes to ml_training
and keeps them out of the honest pool. STEP lines: kind="train" every step.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/train_ps_lopsided.py \
        --size 1b --topk-frac 0.05 --duration 600

MEMORY. Full replica + gradients on every GPU; the server node additionally
holds fp32 master weights and AdamW state for its 1/R chunk. ~15 GB for 1b.
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

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

WIRE = {"bf16": torch.bfloat16, "fp32": torch.float32}


class AdamWChunk:
    """AdamW on one flat chunk, arithmetically identical to torch.optim.AdamW."""

    def __init__(self, p, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = p, lr, *betas, eps, wd
        self.m = torch.zeros_like(p)
        self.v = torch.zeros_like(p)
        self.t = 0

    def step(self, g):
        self.t += 1
        self.p.mul_(1 - self.lr * self.wd)
        self.m.lerp_(g, 1 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        denom = (self.v.sqrt() / math.sqrt(bc2)).add_(self.eps)
        self.p.addcdiv_(self.m, denom, value=-self.lr / bc1)


def pack(vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # int32 indices first: at offset 0 they are always 4-byte aligned, and the
    # values after them start at 4k bytes, aligned for any wire dtype. Values
    # first would put the indices at an odd offset whenever k is odd.
    return torch.cat([idx.to(torch.int32).contiguous().view(torch.uint8),
                      vals.contiguous().view(torch.uint8)])


def unpack(buf: torch.Tensor, k: int, wdt) -> tuple[torch.Tensor, torch.Tensor]:
    return buf[4 * k:].view(wdt).float(), buf[:4 * k].view(torch.int32).long()


def main():
    ap = argparse.ArgumentParser(description="Lopsided parameter server (evasion)")
    ap.add_argument("--size", default="1b", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--duration", type=int, default=600, help="seconds; 0 = no deadline")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (tests)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--topk-frac", type=float, default=0.05,
                    help="c: fraction of gradient entries each worker uploads")
    ap.add_argument("--no-error-feedback", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wire", default="bf16", choices=list(WIRE), help="fp32 for tests")
    ap.add_argument("--backend", default="auto", choices=["auto", "nccl", "gloo"])
    ap.add_argument("--ranks-per-node", type=int, default=0)
    ap.add_argument("--netlog-dir", default="")
    ap.add_argument("--dump-dir", default="")
    args = ap.parse_args()

    topo = Topology(args.backend, args.ranks_per_node)
    if topo.N < 2:
        raise SystemExit("needs at least two nodes: node 0 serves, the others upload")
    dev, wdt = topo.device, WIRE[args.wire]
    wb = torch.tensor([], dtype=wdt).element_size()
    vocab = MODEL_CONFIGS[args.size][4]
    model = build_model(args.size, dev)
    lossf = nn.CrossEntropyLoss()
    amp = dev.type == "cuda"
    server = topo.node == 0

    L = sum(p.numel() for p in model.parameters())
    s, e, C = chunk_bounds(L, topo.R, topo.local)
    k = max(1, math.ceil(args.topk_frac * C))
    theta = padded(flat_params(model), topo.R)
    if server:
        master = theta[s:e].clone()
        opt = AdamWChunk(master, args.lr)
    resid = torch.zeros(C, device=dev)
    ledger = ByteLedger()

    if topo.rank == 0:
        print(f"Lopsided parameter server (ADVERSARIAL training): {args.size}, "
              f"{topo.N} nodes x {topo.R} ranks, c={args.topk_frac}, wire={args.wire}",
              flush=True)
        print(f"  predicted per-node symmetry ~ {args.topk_frac * (wb + 4) / wb:.3f} "
              f"(upload {args.topk_frac * (wb + 4):.2f} B/param, download {wb} B/param)",
              flush=True)

    deadline = CollectiveDeadline(args.duration or None, dev)
    step, t_start = 0, time.time()
    while not deadline.expired():
        if args.max_steps and step >= args.max_steps:
            break
        x, y = synthetic_batch(vocab, args.batch_size, args.seq_len, topo.node, topo.local,
                               step, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        loss = lossf(out.reshape(-1, vocab).float(), y.reshape(-1))
        loss.backward()
        step += 1

        g = padded(torch.cat([p.grad.reshape(-1).float() for p in model.parameters()]),
                   topo.R)
        for p in model.parameters():
            p.grad = None
        if topo.R > 1:
            dist.all_reduce(g, group=topo.intra)
            g.div_(topo.R)
        gc = g[s:e]

        if server:
            total = gc.clone()
            buf = torch.empty(k * (wb + 4), dtype=torch.uint8, device=dev)
            for n in range(1, topo.N):
                dist.recv(buf, src=topo.peer(n))
                ledger.add(step, recv=buf.numel())
                vals, idx = unpack(buf, k, wdt)
                total.index_add_(0, idx, vals)
            opt.step(total / topo.N)
            wire = master.to(wdt)
            wbuf = wire.contiguous().view(torch.uint8)
            for n in range(1, topo.N):
                dist.send(wbuf, dst=topo.peer(n))
                ledger.add(step, sent=wbuf.numel())
            mine = wire.float()
        else:
            ef = gc + resid if not args.no_error_feedback else gc
            idx = torch.topk(ef.abs(), k, sorted=False).indices
            vals = ef[idx]
            if not args.no_error_feedback:
                resid = ef.clone()
                resid[idx] = 0
            buf = pack(vals.to(wdt), idx)
            dist.send(buf, dst=topo.peer(0))
            ledger.add(step, sent=buf.numel())
            wbuf = torch.empty(C * wb, dtype=torch.uint8, device=dev)
            dist.recv(wbuf, src=topo.peer(0))
            ledger.add(step, recv=wbuf.numel())
            mine = wbuf.view(wdt).float()

        if topo.R > 1:
            parts = [torch.empty_like(mine) for _ in range(topo.R)]
            dist.all_gather(parts, mine.contiguous(), group=topo.intra)
            theta = torch.cat(parts)
        else:
            theta = mine
        load_flat(model, theta[:L])

        if topo.rank == 0:
            emit_step("train", step, tokens=args.batch_size * args.seq_len, passes=1,
                      node=topo.node, rank=topo.rank, method="ps_lopsided",
                      c=args.topk_frac)

    el = time.time() - t_start
    extra = dict(steps=step, elapsed_s=round(el, 2), role="server" if server else "worker",
                 topk_frac=args.topk_frac, wire=args.wire, params=L,
                 predicted_symmetry=round(args.topk_frac * (wb + 4) / wb, 4))
    ledger.dump(topo, args.netlog_dir or None, extra)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        torch.save(dict(params=flat_params(model).cpu(), steps=step),
                   os.path.join(args.dump_dir, f"rank{topo.rank}.pt"))
    if topo.rank == 0:
        print(f"PS done: {step} steps, {el:.0f}s", flush=True)
    sync_teardown()


if __name__ == "__main__":
    main()
