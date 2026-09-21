"""What each evasion costs the adversary.

A detector that only catches evasions nobody can afford has proven nothing, so
every evasion needs a price tag next to its volume reduction. Three of the four
prices here are computable in closed form. The fourth -- what diluting gradient
synchronisation does to CONVERGENCE -- is not, and is left explicitly blank
rather than guessed at.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import commvol as C

BASE = dict(model="Llama-3-70B", hw="H100-SXM", dp=64, tp=8, pp=1,
            global_batch=2048, seq=4096, sharding="fsdp")

def baseline():
    m, hw = C.MODELS[BASE["model"]], C.HARDWARE[BASE["hw"]]
    return C.training_comm(m, hw, dp=BASE["dp"], tp=BASE["tp"], pp=BASE["pp"],
                           global_batch=BASE["global_batch"], seq=BASE["seq"],
                           sharding=BASE["sharding"])

def shaped(cap_frac: float):
    """Traffic shaping: hold interconnect utilisation to `cap_frac` of the link.
    The bytes still have to move, so the step simply takes longer."""
    m, hw = C.MODELS[BASE["model"]], C.HARDWARE[BASE["hw"]]
    capped = C.HW(hw.name, hw.flops_bf16, hw.hbm_bw, hw.hbm_bytes,
                  hw.nvlink_bw*cap_frac, hw.nic_bw*cap_frac, hw.gpus_per_node)
    return C.training_comm(m, capped, dp=BASE["dp"], tp=BASE["tp"], pp=BASE["pp"],
                           global_batch=BASE["global_batch"], seq=BASE["seq"],
                           sharding=BASE["sharding"])

def segmented(n_segments: int, restart_min: float = 12.0, total_h: float = 720.0):
    """Run segmentation: chop a long run into pieces to stay under a reporting
    threshold. Cost is checkpoint save + reload + warm-up per restart."""
    overhead_h = n_segments * restart_min / 60.0
    return dict(evasion="run segmentation", knob=n_segments,
                volume_ratio=1.0,                    # per-step traffic unchanged
                inter_ratio=1.0, throughput_retained=total_h/(total_h+overhead_h),
                note="per-step traffic identical; only the run BOUNDARY moves",
                convergence_cost="none")

rows = []
b = baseline()
b_rate, b_step, b_inter, b_intra = b.total_bps, b.step_time_s, b.inter_bps, b.intra_bps
hw0 = C.HARDWARE[BASE["hw"]]
print(f"baseline: {BASE['model']} FSDP dp64 tp8 on H100")
print(f"  {b_rate/1e9:6.0f} GB/s per GPU total  =  {b_intra/1e9:.0f} intra (NVLink) "
      f"+ {b_inter/1e9:.1f} inter (NIC)")
print(f"  step {b_step:.2f}s, comm-bound={b.meta['comm_bound']}, "
      f"effective MFU {b.meta['mfu_effective']:.3f}")
print(f"  link utilisation: NVLink {b_intra/hw0.nvlink_bw:.0%}, NIC {b_inter/hw0.nic_bw:.0%}")
print("  -> tensor-parallel traffic dominates, and it is intra-node. An evasion")
print("     that only touches the DATA-parallel sync barely moves the total, but")
print("     it is the INTER-node number our monitor would be reading.\n")
rec = lambda **kw: rows.append(kw)

for h in (1, 10, 50, 100, 500, 1000):
    m, hw = C.MODELS[BASE["model"]], C.HARDWARE[BASE["hw"]]
    w = C.training_comm(m, hw, dp=BASE["dp"], tp=BASE["tp"], pp=BASE["pp"],
                        global_batch=BASE["global_batch"], seq=BASE["seq"],
                        sharding="diloco", diloco_inner=h)
    rec(evasion="DiLoCo", knob=h, volume_ratio=w.total_bps/b_rate,
        inter_ratio=w.inter_bps/max(b_inter,1), throughput_retained=b_step/w.step_time_s,
        convergence_cost="NOT MODELLED - must be measured empirically")
for k in (1, 2, 4, 8, 16, 32):
    m, hw = C.MODELS[BASE["model"]], C.HARDWARE[BASE["hw"]]
    w = C.training_comm(m, hw, dp=BASE["dp"], tp=BASE["tp"], pp=BASE["pp"],
                        global_batch=BASE["global_batch"], seq=BASE["seq"],
                        sharding=BASE["sharding"], grad_accum=k)
    rec(evasion="gradient accumulation", knob=k, volume_ratio=w.total_bps/b_rate,
        inter_ratio=w.inter_bps/max(b_inter,1), throughput_retained=b_step*k/w.step_time_s,
        convergence_cost="larger effective batch")
for c in (1.0, 0.75, 0.5, 0.25, 0.1, 0.05):
    w = shaped(c)
    rec(evasion="traffic shaping", knob=c, volume_ratio=w.total_bps/b_rate,
        inter_ratio=w.inter_bps/max(b_inter,1), throughput_retained=b_step/w.step_time_s,
        convergence_cost="none")
for n in (1, 2, 4, 8, 16, 32):
    rows.append(segmented(n))

df = pd.DataFrame(rows)
df.to_csv("evasion_cost.csv", index=False)
pd.set_option("display.width", 200)
print()
print(df[["evasion","knob","volume_ratio","inter_ratio","throughput_retained"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

# the same evasion against a baseline WITHOUT tensor parallelism, where the
# data-parallel sync is the only traffic there is
m, hw = C.MODELS["Llama-2-7B"], C.HARDWARE["H100-SXM"]
d0 = C.training_comm(m, hw, dp=512, tp=1, pp=1, global_batch=2048, seq=2048)
d5 = C.training_comm(m, hw, dp=512, tp=1, pp=1, global_batch=2048, seq=2048,
                     sharding="diloco", diloco_inner=500)
print(f"\nDiLoCo(500) against a pure-DDP 7B baseline (no TP traffic to hide behind):")
print(f"  volume  {d5.total_bps/d0.total_bps:.4f} of baseline "
      f"({d0.total_bps/1e9:.0f} -> {d5.total_bps/1e9:.2f} GB/s per GPU)")
print(f"  throughput retained {d0.step_time_s/d5.step_time_s:.2f}x "
      f"(baseline comm-bound: {d0.meta['comm_bound']})")

# the same evasion where the baseline IS link-limited: large dp, small local batch
c0 = C.training_comm(m, hw, dp=1024, tp=1, pp=1, global_batch=1024, seq=2048)
c5 = C.training_comm(m, hw, dp=1024, tp=1, pp=1, global_batch=1024, seq=2048,
                     sharding="diloco", diloco_inner=500)
print(f"\nSame evasion where the baseline is COMMUNICATION-BOUND "
      f"(dp=1024, one sequence per GPU):")
print(f"  baseline comm-bound: {c0.meta['comm_bound']}, "
      f"effective MFU {c0.meta['mfu_effective']:.3f} of a nominal 0.45")
print(f"  volume  {c5.total_bps/c0.total_bps:.4f} of baseline")
print(f"  throughput retained {c0.step_time_s/c5.step_time_s:.2f}x  <-- the adversary")
print(f"     goes FASTER: it was waiting on the network and now it is not.")
print("\nSo the cost of DiLoCo in wall-clock terms is between zero and negative.")
print("Its only real price is a convergence penalty, which is NOT computable here")
print("and must be measured. Until it is, we cannot claim this evasion is costly.")
