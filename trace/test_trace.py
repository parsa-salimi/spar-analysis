"""Round-trip and safety tests for comm_trace/1.

These are the assertions that make the synthetic traces trustworthy enough to
build a detector on. If they fail, anything downstream is decoration.
"""
from __future__ import annotations
import sys, tempfile, shutil
import config  # noqa: F401  (puts the repo on sys.path)

import numpy as np, pandas as pd
import commvol as C
from commtrace.gen import synth_trace, Topology, Collection
from commtrace.reader import rates, window_features, preflight
from commtrace.schema import read_trace, write_trace, Link, Manifest

results = []
def check(ok, what, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail else ""))
    results.append(bool(ok))

TRAIN = C.training_comm(C.MODELS["Llama-3-70B"], C.HARDWARE["H100-SXM"],
                        dp=8, tp=8, global_batch=512, seq=4096, sharding="fsdp")
INFER = C.inference_comm(C.MODELS["Llama-2-7B"], C.HARDWARE["H100-SXM"],
                         mode="disaggregated", tp=8, replicas=4, batch=64,
                         prompt_len=2048, output_len=256)
tmp = tempfile.mkdtemp()

# ── 1. round trip: what the reader recovers is what the generator laid down ──
p, truth = synth_trace(TRAIN, f"{tmp}/rt", duration_s=120,
                       coll=Collection(sample_hz=10, nvlink_width=64, ib_width=64))
r = rates(p)
rec = r.groupby("link_key")[["tx_bytes", "rx_bytes"]].sum()
ratios = []
for key, row in rec.iterrows():
    for d in ("tx", "rx"):
        t = truth[f"{key}/{d}"]
        if t > 0:
            ratios.append(row[f"{d}_bytes"] / t)
ratios = np.array(ratios)
check(bool(((ratios > 0.95) & (ratios <= 1.001)).all()),
      "recovered bytes match the generator's ground truth",
      f"ratio {ratios.min():.4f}-{ratios.max():.4f} over {len(ratios)} counters")

# ── 2. the wrap check fires when the counter is too narrow for the rate ─────
narrow = Collection(sample_hz=10, nvlink_unit="words4", nvlink_width=32)
pn, truth_n = synth_trace(TRAIN, f"{tmp}/narrow", duration_s=60, coll=narrow)
rn = rates(pn)
nv = rn[rn.link_type == "nvlink"]
wrap_s = float(nv.wrap_seconds.iloc[0])
check(float(nv.wrap_rate_exceeded.mean()) == 0.0,
      "the RATE heuristic is blind to multi-wrap, as predicted",
      f"0% flagged by rate, though the counter wraps every {wrap_s*1000:.0f} ms "
      "and is sampled every 100 ms")
check(float(nv.wrap_undersampled.mean()) == 1.0,
      "the CONFIG check catches it -- sample interval exceeds the wrap period",
      f"100% of intervals flagged")
pf = preflight(pn)
check(not pf[pf.link_type == "nvlink"].safe.any(),
      "preflight refuses the link before a single value is read",
      f"min safe rate {pf[pf.link_type=='nvlink'].min_safe_hz.iloc[0]:,.0f} Hz "
      "vs 10 Hz configured")
wide = Collection(sample_hz=10, nvlink_unit="bytes", nvlink_width=64,
                  ib_unit="words4", ib_width=64)
pw, _ = synth_trace(TRAIN, f"{tmp}/wide", duration_s=60, coll=wide)
check(float(rates(pw).wrap_suspect.mean()) == 0.0,
      "64-bit counters on the same traffic are not flagged")
# The DEFAULT InfiniBand configuration -- the standard 32-bit sysfs word
# counters -- is itself unsafe, which is worth asserting rather than assuming.
pdef, _ = synth_trace(TRAIN, f"{tmp}/ibdefault", duration_s=30,
                      coll=Collection(sample_hz=10, nvlink_width=64))
pf_ib = preflight(pdef)
pf_ib = pf_ib[pf_ib.link_type == "ib"]
check(not pf_ib.safe.any(),
      "the STANDARD 32-bit sysfs IB counters are unsafe at 10 Hz",
      f"needs {pf_ib.min_safe_hz.iloc[0]:,.0f} Hz; use the 64-bit extended counters")

# ── 3. and the narrow counter really is WRONG, not merely suspicious ────────
bad = rn[rn.link_type == "nvlink"].groupby("link_key").tx_bytes.sum()
err = np.array([bad[k] / truth_n[f"{k}/tx"] for k in bad.index])
check(bool((err < 0.5).all()), "un-flagged, the narrow counter would under-report badly",
      f"recovers only {err.mean():.1%} of the true bytes")

# ── 4. symmetry separates training from disaggregated serving at NODE level ─
pt, _ = synth_trace(TRAIN, f"{tmp}/sym_t", duration_s=120,
                    coll=Collection(sample_hz=10, ib_width=64, nvlink_width=64))
pi, _ = synth_trace(INFER, f"{tmp}/sym_i", duration_s=120, topo=Topology(n_nodes=4),
                    coll=Collection(sample_hz=10, ib_width=64, nvlink_width=64))
st = window_features(rates(pt)); si = window_features(rates(pi))
st_ib = st[st.link_type == "ib"].symmetry.mean()
si_ib = si[si.link_type == "ib"].symmetry.mean()
check(st_ib > 0.9 and si_ib < 0.2,
      "inter-node symmetry: training ~1, disaggregated serving ~0",
      f"training {st_ib:.3f} vs serving {si_ib:.3f}")
si_nv = si[si.link_type == "nvlink"].symmetry.mean()
check(si_nv > 0.9, "…while that SAME serving trace is symmetric on NVLink",
      f"{si_nv:.3f} -- tensor-parallel decode is an all-reduce, so symmetry is "
      "not a class label in disguise")

# ── 5. the schema makes the summed-bytes mistake unrepresentable ────────────
obs, links, man = read_trace(pt)
try:
    obs2 = obs.copy(); obs2["total"] = 1
    write_trace(f"{tmp}/bad", obs2, links, man)
    check(False, "writer rejects a summed-bytes column")
except AssertionError:
    check(True, "writer rejects a summed-bytes column")

# ── 6. the label cannot leak into a feature matrix ──────────────────────────
check("label" not in obs.columns and man.ground_truth is not None,
      "ground truth lives in the manifest, never in observations.parquet")
p_unl, _ = synth_trace(TRAIN, f"{tmp}/unlabelled", duration_s=30, with_label=False)
check(read_trace(p_unl)[2].ground_truth is None,
      "an unlabelled trace carries no ground truth at all")

# ── 7. unwrap arithmetic is right across an actual rollover ─────────────────
w = 32; mask = np.uint64((1 << w) - 1)
v1 = np.uint64(2**32 - 10); v2 = np.uint64(5)          # wrapped by 15
check(int(np.uint64(int(v2) - int(v1) + 2**w) & mask) == 15,
      "modular unwrap is correct across a rollover")

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{sum(results)}/{len(results)} checks pass")
sys.exit(0 if all(results) else 1)
