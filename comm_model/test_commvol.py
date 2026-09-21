"""Anchor tests.

Each one pins the model to a number that can be derived by hand, so a future
edit that breaks the arithmetic fails loudly instead of quietly producing a
plausible-looking plot.
"""
import math, sys
import commvol as C

def close(a, b, tol=0.02, what=""):
    ok = abs(a-b) <= tol*max(abs(a), abs(b), 1e-30)
    print(f"{'PASS' if ok else 'FAIL'}  {what}: {a:.6g} vs {b:.6g}")
    return ok

results = []
M, HW = C.MODELS, C.HARDWARE
m7, m8 = M["Llama-2-7B"], M["Llama-3-8B"]
h100 = HW["H100-SXM"]

# 1. ring all-reduce factor is 2(N-1)/N, NOT 2
results.append(close(C.f_allreduce(2), 1.0, 0, "ring factor at N=2 is 1.0 not 2.0"))
results.append(close(C.f_allreduce(8), 1.75, 0, "ring factor at N=8"))
results.append(close(C.f_allreduce(1024), 2*1023/1024, 0, "ring factor -> 2 for large N"))

# 2. DDP moves 2*Psi*b per GPU per step (large N)
w = C.training_comm(m7, h100, dp=1024, global_batch=1024)
bps = w.total_bps * w.step_time_s
results.append(close(bps, C.f_allreduce(1024)*m7.params*2, 1e-9, "DDP 7B bytes/step/GPU"))
print(f"      -> {bps/1e9:.1f} GB/step/GPU (hand figure: 2*6.74e9*2 = 26.9 GB)")

# 3. FSDP/ZeRO-3 is exactly 1.5x DDP (3 passes vs 2), without checkpointing
w2 = C.training_comm(m7, h100, dp=1024, sharding="fsdp", global_batch=1024)
per_step = lambda x: x.total_bps * x.step_time_s
results.append(close(per_step(w2)/per_step(w), 1.5, 1e-9, "FSDP / DDP bytes per STEP"))
# ...but at this scale both are pinned at NIC bandwidth, so the RATES are equal:
# FSDP moves 1.5x the bytes and takes 1.5x as long to move them.
results.append(close(w2.total_bps/w.total_bps, 1.0, 1e-6,
                     "FSDP / DDP bytes per SECOND (both communication-bound)"))
results.append(close(float(w.meta["comm_bound"]), 1.0, 0,
                     "7B DDP at dp=1024 is communication-bound"))
print(f"      -> effective MFU collapses to {w.meta['mfu_effective']:.3f} from 0.45")
w3 = C.training_comm(m7, h100, dp=1024, sharding="fsdp", global_batch=1024,
                     checkpointing=True)
results.append(close(per_step(w3)/per_step(w), 2.0, 1e-9,
                     "FSDP+checkpointing / DDP bytes per step"))

# 4. tensor parallel: 4 all-reduces per layer, activation-sized
w4 = C.training_comm(m7, h100, dp=1, tp=8, global_batch=8, seq=2048)
results.append(close(w4.collectives_per_step, 4*32, 0, "TP collectives/step (4 x 32 layers)"))
results.append(close(w4.msg_bytes, 8*2048*4096*2, 1e-9, "TP message size"))
print(f"      -> {w4.msg_bytes/1e6:.1f} MB per all-reduce, "
      f"{w4.collectives_per_step*w4.msg_bytes/1e9:.1f} GB of buffers/step, "
      f"{w4.total_bps*w4.step_time_s/1e9:.1f} GB actually on the wire")

# 5. KV cache per token, and what GQA buys
results.append(close(m7.kv_bytes_per_token(), 512*1024, 0, "Llama-2-7B KV/token (MHA)"))
results.append(close(m8.kv_bytes_per_token(), 128*1024, 0, "Llama-3-8B KV/token (GQA)"))
results.append(close(m7.kv_bytes_per_token()/m8.kv_bytes_per_token(), 4.0, 0,
                     "GQA reduction factor (32 KV heads -> 8)"))

# 6. NVLink is ~9x a NIC per GPU on H100 -- why TP must stay inside a node
results.append(close(h100.nvlink_bw/h100.nic_bw, 9.0, 0, "NVLink : NIC per-GPU ratio"))

# 7. disaggregated 7B at ~25 req/s with 2048-token prompts is ~25 GB/s aggregate
qps, plen = 25.0, 2048
agg = qps * plen * m7.kv_bytes_per_token()
results.append(close(agg, 26.8e9, 0.02, "KV push aggregate at 25 req/s"))
print(f"      -> {agg/1e9:.1f} GB/s of KV traffic, against ~25-27 GB/s for 7B DDP")

# 8. the two volume-reduction evasions scale exactly as advertised
wd = C.training_comm(m7, h100, dp=1024, sharding="diloco", global_batch=1024,
                     diloco_inner=500)
results.append(close(per_step(w)/per_step(wd), 500.0, 1e-6,
                     "DiLoCo(500) bytes-per-STEP reduction"))
# The RATE reduction is far smaller, because the DDP baseline it escapes was
# already pinned at link speed while DiLoCo is back to compute-bound. The gap is
# throughput the adversary GAINS -- this evasion has a negative cost.
results.append(close(w.total_bps/wd.total_bps, 172.73, 0.01,
                     "DiLoCo(500) byte-RATE reduction (smaller: DDP was comm-bound)"))
print(f"      -> DiLoCo step {wd.step_time_s:.3f}s vs DDP {w.step_time_s:.3f}s = "
      f"{w.step_time_s/wd.step_time_s:.2f}x FASTER for the adversary")
wa = C.training_comm(m7, h100, dp=1024, global_batch=1024, grad_accum=8)
results.append(close(per_step(w)/(per_step(wa)*8), 1.0, 1e-6,
                     "grad-accum(8) cuts per-token sync volume 8x"))

# 9. symmetry and size variability come from the FLOWS, not from the class label
tp_infer = C.inference_comm(m7, h100, mode="colocated", tp=8)
pp_train = C.training_comm(M["Llama-3-70B"], h100, dp=1, tp=1, pp=8, global_batch=8)
results.append(close(tp_infer.sym, 1.0, 0, "colocated TP INFERENCE is symmetric"))
results.append(close(pp_train.sym, 0.0, 0, "pipeline TRAINING is directional"))
print("      -> symmetry cross-cuts the classes, so it is not a circular feature")

print(f"\n{sum(results)}/{len(results)} anchors pass")
sys.exit(0 if all(results) else 1)
