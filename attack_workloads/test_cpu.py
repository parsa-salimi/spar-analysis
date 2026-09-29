#!/usr/bin/env python3
"""CPU tests for the attack workloads: gloo backend, processes pretending to be nodes.

What is checked
  maths      DiLoCo's outer step and the parameter server's update match
             independent single-process reference implementations exactly
             (fp32 wire, so no rounding hides a bug).
  agreement  every rank ends with identical weights -- inner DDP, the outer
             exchange and the parameter download keep all replicas in step.
  signature  the byte ledger shows the network shape each attack is meant to
             have: DiLoCo silent between syncs, bursty or spread; the parameter
             server lopsided at the predicted symmetry.
  shutdown   a timed run ends cleanly on the shared deadline.

    python3 test_cpu.py            # from this directory
"""
import json, math, os, shutil, subprocess, sys, tempfile, glob

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(HERE, "robi_stub")
# The workers run with OMP_NUM_THREADS=1. The references must too: Adam's first
# update is ~lr * sign(gradient), so a near-zero gradient entry whose low bits
# differ between thread counts flips the update by 2 * lr.
torch.set_num_threads(1)
sys.path[:0] = [HERE, STUB]
from _mn_attack import MODEL_CONFIGS, build_model, synthetic_batch, flat_params  # noqa: E402
import train_diloco as D  # noqa: E402

TINY = ["--size", "tiny", "--batch-size", "2", "--seq-len", "8", "--backend", "gloo"]
RESULTS = []


def run(script, nproc, rpn, extra, tmp):
    env = {**os.environ, "PYTHONPATH": f"{HERE}:{STUB}", "OMP_NUM_THREADS": "1"}
    cmd = ["torchrun", "--standalone", f"--nproc-per-node={nproc}", os.path.join(HERE, script),
           *TINY, "--ranks-per-node", str(rpn), "--netlog-dir", os.path.join(tmp, "net"),
           "--dump-dir", os.path.join(tmp, "dump"), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"{script} {extra} exited {p.returncode}")
    nets = {}
    for f in glob.glob(os.path.join(tmp, "net", "*.json")):
        d = json.load(open(f)); nets[d["rank"]] = d
    dumps = {int(os.path.basename(f)[4:-3]): torch.load(f)
             for f in glob.glob(os.path.join(tmp, "dump", "rank*.pt"))}
    return nets, dumps, p.stdout


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def node_sym(nets, rpn):
    out = {}
    for r, d in nets.items():
        n = r // rpn
        tx, rx = out.get(n, (0, 0))
        out[n] = (tx + d["tx_bytes"], rx + d["rx_bytes"])
    return {n: (min(t, x) / max(t, x) if max(t, x) else 1.0, t, x) for n, (t, x) in out.items()}


def all_equal(dumps, key="params", tol=0.0):
    ps = [dumps[r][key] for r in sorted(dumps)]
    return max(float((p - ps[0]).abs().max()) for p in ps) <= tol


# ── 1. DiLoCo outer step vs a reference, 2 nodes x 1 rank, fp32 wire ────────
def t_diloco_math():
    H = 3
    with tempfile.TemporaryDirectory() as tmp:
        _, dumps, _ = run("train_diloco.py", 2, 1, ["--inner-steps", str(H), "--max-steps",
                          str(H), "--duration", "0", "--outer-quant", "fp32"], tmp)
    vocab = MODEL_CONFIGS["tiny"][4]
    A = flat_params(build_model("tiny", "cpu"))
    deltas = []
    for node in (0, 1):
        m = build_model("tiny", "cpu")
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        for step in range(H):
            x, y = synthetic_batch(vocab, 2, 8, node, 0, step, "cpu")
            nn.CrossEntropyLoss()(m(x).reshape(-1, vocab), y.reshape(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        deltas.append(A - flat_params(m))
    avg = torch.stack(deltas).mean(0)
    mom = avg.clone()
    G = A - 0.7 * (avg + 0.9 * mom)
    err = float((dumps[0]["G"] - G).abs().max())
    check("DiLoCo outer Nesterov step matches a reference implementation", err < 1e-6,
          f"max |diff| {err:.1e}")
    check("DiLoCo: both nodes hold the new global model after the sync",
          all_equal(dumps) and float((dumps[0]["params"] - G).abs().max()) < 1e-6)


# ── 2. DiLoCo burst, 2 nodes x 2 ranks: silence between syncs ───────────────
def t_diloco_burst():
    H, S = 4, 8
    with tempfile.TemporaryDirectory() as tmp:
        nets, dumps, out = run("train_diloco.py", 4, 2, ["--inner-steps", str(H),
                               "--max-steps", str(S), "--duration", "0"], tmp)
    steps = sorted({int(k) for d in nets.values() for k, v in d["per_step"].items() if v[0] or v[1]})
    check("DiLoCo burst: cross-node traffic only on sync steps", steps == [4, 8], f"steps {steps}")
    check("DiLoCo burst: all four ranks agree after the last sync", all_equal(dumps))
    L = nets[0]["params"]
    per_node = sum(nets[r]["tx_bytes"] for r in (0, 1)) / 2        # per sync
    exp = L * 2
    check("DiLoCo burst: each node sends one bf16 model per sync, not R",
          abs(per_node - exp) / exp < 0.01, f"{per_node:.0f} B vs {exp} B")
    sym = node_sym(nets, 2)
    check("DiLoCo burst: every node balanced (it is still an all-gather)",
          all(v[0] > 0.999 for v in sym.values()))


# ── 3. DiLoCo spread: the sync becomes a trickle ────────────────────────────
def t_diloco_spread():
    H, S = 4, 12
    with tempfile.TemporaryDirectory() as tmp:
        nets, dumps, _ = run("train_diloco.py", 4, 2, ["--inner-steps", str(H), "--max-steps",
                             str(S), "--duration", "0", "--schedule", "spread"], tmp)
    ps = nets[0]["per_step"]
    steps = sorted(int(k) for k, v in ps.items() if v[0])
    sizes = {ps[str(k)][0] for k in steps}
    check("DiLoCo spread: one equal piece on every step after the first sync",
          steps == list(range(5, 13)) and len(sizes) == 1, f"steps {steps[:3]}..{steps[-1]}, sizes {sizes}")
    same_node = all(float((dumps[2 * n]["params"] - dumps[2 * n + 1]["params"]).abs().max()) == 0
                    for n in (0, 1))
    check("DiLoCo spread: ranks within a node agree", same_node)


# ── 4. DiLoCo wire formats ──────────────────────────────────────────────────
def t_quant():
    x = torch.randn(10_000)
    for fmt, tol in (("none", 1e-2), ("int8", 2e-2), ("int4", 3e-1), ("fp32", 0.0)):
        y = D.decode(D.encode(x, fmt), fmt, x.numel())
        rel = float((y - x).abs().max() / x.abs().max())
        check(f"wire format {fmt}: round trip within {tol} of max", rel <= tol, f"{rel:.3g}")
    import train_ps_lopsided as P
    ok = True
    for k in (1, 2, 3, 7, 1000):
        for wdt in (torch.bfloat16, torch.float32):
            v, i = torch.randn(k).to(wdt), torch.randint(0, 10**6, (k,))
            v2, i2 = P.unpack(P.pack(v, i), k, wdt)
            ok &= torch.equal(v2, v.float()) and torch.equal(i2, i)
    check("PS upload packing round-trips for odd and even k, bf16 and fp32", ok)
    with tempfile.TemporaryDirectory() as tmp:
        nets, dumps, _ = run("train_diloco.py", 4, 2, ["--inner-steps", "2", "--max-steps", "2",
                             "--duration", "0", "--outer-quant", "int4"], tmp)
    L = nets[0]["params"]
    per_node = sum(nets[r]["tx_bytes"] for r in (0, 1))
    check("DiLoCo int4: ~0.5 bytes/param on the wire", per_node < 0.6 * L,
          f"{per_node / L:.3f} B/param")
    check("DiLoCo int4: all ranks agree after the sync", all_equal(dumps))


# ── 5. Parameter server maths, c = 1, fp32 wire: must equal plain DP + AdamW ─
def t_ps_math():
    S = 3
    with tempfile.TemporaryDirectory() as tmp:
        _, dumps, _ = run("train_ps_lopsided.py", 2, 1, ["--topk-frac", "1.0", "--wire", "fp32",
                          "--max-steps", str(S), "--duration", "0"], tmp)
    vocab = MODEL_CONFIGS["tiny"][4]
    m = build_model("tiny", "cpu")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for step in range(S):
        gs = []
        for node in (0, 1):
            m.zero_grad(set_to_none=True)
            x, y = synthetic_batch(vocab, 2, 8, node, 0, step, "cpu")
            nn.CrossEntropyLoss()(m(x).reshape(-1, vocab), y.reshape(-1)).backward()
            gs.append([p.grad.clone() for p in m.parameters()])
        for p, g0, g1 in zip(m.parameters(), *gs):
            p.grad = (g0 + g1) / 2
        opt.step()
    ref = flat_params(m)
    err = max(float((dumps[r]["params"] - ref).abs().max()) for r in dumps)
    check("PS with c=1 equals data-parallel AdamW on the averaged gradient", err < 1e-6,
          f"max |diff| {err:.1e}")


# ── 6. Parameter server, 3 nodes x 2 ranks, c = 0.1: the lopsided signature ─
def t_ps_lopsided():
    c = 0.1
    with tempfile.TemporaryDirectory() as tmp:
        nets, dumps, out = run("train_ps_lopsided.py", 6, 2, ["--topk-frac", str(c),
                               "--max-steps", "5", "--duration", "0"], tmp)
    sym = node_sym(nets, 2)
    pred = c * 3
    ok = all(abs(v[0] - pred) < 0.02 for v in sym.values())
    detail = ", ".join(f"node{n} {v[0]:.3f} ({'tx>rx' if v[1] > v[2] else 'rx>tx'})"
                       for n, v in sorted(sym.items()))
    check(f"PS c={c}: every node lopsided at the predicted symmetry {pred:.2f}", ok, detail)
    check("PS: server sends more than it receives, workers the reverse",
          sym[0][1] > sym[0][2] and all(sym[n][2] > sym[n][1] for n in (1, 2)))
    check("PS: every rank on every node holds the same weights", all_equal(dumps))
    steps = sorted(int(k) for k, v in nets[0]["per_step"].items() if v[0] or v[1])
    check("PS: traffic every step (it is not a rare-sync attack)", steps == [1, 2, 3, 4, 5])


# ── 7. Timed runs end cleanly on the shared deadline ────────────────────────
def t_deadline():
    for script in ("train_diloco.py", "train_ps_lopsided.py"):
        with tempfile.TemporaryDirectory() as tmp:
            extra = ["--duration", "4"] + (["--inner-steps", "5"] if "diloco" in script else [])
            nets, dumps, out = run(script, 4, 2, extra, tmp)
        steps = {d["steps"] for d in nets.values()}
        check(f"{script}: 4 s timed run exits 0, every rank stops on the same step",
              len(steps) == 1 and len(dumps) == 4, f"steps {steps}")


if __name__ == "__main__":
    for t in (t_quant, t_diloco_math, t_diloco_burst, t_diloco_spread, t_ps_math,
              t_ps_lopsided, t_deadline):
        t()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks pass")
    sys.exit(0 if all(RESULTS) else 1)
