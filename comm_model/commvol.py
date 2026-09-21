"""Closed-form interconnect communication for training and inference workloads.

One question: for a given workload configuration, how many bytes cross the
interconnect per GPU per second, and with what *structure*?

Design note — why traffic is modelled as COMPONENTS
---------------------------------------------------
An earlier version of this file attached a symmetry and a cadence value to each
workload according to its class ("training is metronomic, inference is Poisson").
That is circular: any classifier fed those features would separate the classes
perfectly because I had already separated them by hand.

So every workload is instead built from `Flow` components, each of which is a
specific collective or transfer with properties that follow from ITS OWN
mechanism, not from the workload's label:

    ring all-reduce   symmetric (sends == receives), fixed size
    all-gather        symmetric, fixed size
    reduce-scatter    symmetric, fixed size
    all-to-all        symmetric, size varies with token routing
    pipeline p2p      DIRECTIONAL, fixed size
    KV-cache push     DIRECTIONAL, size varies with prompt length

The aggregate symmetry and size-variability of a workload then fall out of
which flows it happens to contain. These genuinely cross-cut the classes:
tensor-parallel *inference* is all-reduce traffic and therefore symmetric, while
pipeline-parallel *training* is directional. That is what makes a separability
result from these features meaningful rather than assumed.

Byte conventions
----------------
Returns are bytes actually moved on the wire per GPU, not buffer sizes. Ring
all-reduce moves 2(N-1)/N * M per GPU, NOT 2M — at N=2 the factor is 1.0, so the
common `* 2` over-counts two-GPU runs by exactly 2x.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ── collective cost factors: bytes on the wire per GPU, per unit buffer ─────
def f_allreduce(n: int) -> float:
    """Ring all-reduce = reduce-scatter + all-gather = 2(N-1)/N per GPU."""
    return 0.0 if n <= 1 else 2.0 * (n - 1) / n

def f_allgather(n: int) -> float:
    return 0.0 if n <= 1 else (n - 1) / n

def f_reducescatter(n: int) -> float:
    return 0.0 if n <= 1 else (n - 1) / n

def f_alltoall(n: int) -> float:
    return 0.0 if n <= 1 else (n - 1) / n


# ── properties that belong to a MECHANISM, not to a workload class ──────────
#   sym      : 1.0 = sends what it receives; 0.0 = strictly one-directional
#   size_cv  : coefficient of variation of individual transfer sizes
# Step-driven traffic fires on a loop boundary, whether that loop is a training
# step or a decode step. There is no principled reason to give them different
# jitter, and giving them different constants would make cadence a disguised
# class label. One value for both; only request-driven traffic differs.
STEP_CV = 0.03
REQUEST_CV = 1.0

FLOW_KINDS = {
    "allreduce":     dict(sym=1.0, size_cv=0.00),
    "allgather":     dict(sym=1.0, size_cv=0.00),
    "reducescatter": dict(sym=1.0, size_cv=0.00),
    "alltoall":      dict(sym=1.0, size_cv=0.35),   # expert routing is imbalanced
    "p2p":           dict(sym=0.0, size_cv=0.00),   # pipeline stage -> stage
    "kv_push":       dict(sym=0.0, size_cv=0.80),   # scales with prompt length
}


@dataclass
class Flow:
    kind: str
    bytes_per_s: float
    msg_bytes: float
    count_per_step: float
    fabric: str            # "intra" (NVLink) | "inter" (NIC)
    interval_cv: float     # CV of the gap between successive transfers

    @property
    def sym(self) -> float:      return FLOW_KINDS[self.kind]["sym"]
    @property
    def size_cv(self) -> float:  return FLOW_KINDS[self.kind]["size_cv"]


@dataclass
class HW:
    name: str
    flops_bf16: float      # dense bf16 FLOP/s per GPU
    hbm_bw: float          # bytes/s per GPU
    hbm_bytes: float       # memory capacity per GPU
    nvlink_bw: float       # bytes/s per GPU per direction
    nic_bw: float          # bytes/s per GPU per direction
    gpus_per_node: int = 8

HARDWARE = {
    "A100-80GB": HW("A100-80GB", 312e12, 2.039e12,  80e9, 300e9, 25e9),
    "H100-SXM":  HW("H100-SXM",  989e12, 3.35e12,   80e9, 450e9, 50e9),
    "H200-SXM":  HW("H200-SXM",  989e12, 4.8e12,   141e9, 450e9, 50e9),
    "B200":      HW("B200",     2250e12, 8.0e12,   180e9, 900e9, 50e9),
}


@dataclass(frozen=True)
class Model:
    name: str
    params: float
    layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    d_head: int = 128
    active_params: float | None = None
    n_experts: int = 0
    top_k: int = 0
    moe_layers: int = 0

    @property
    def active(self) -> float:
        return self.active_params or self.params

    def kv_bytes_per_token(self, b: int = 2) -> float:
        """K and V, every layer, every KV head. GQA cuts this by n_heads/n_kv_heads."""
        return 2 * self.layers * self.n_kv_heads * self.d_head * b

MODELS = {
    "Llama-2-7B":   Model("Llama-2-7B",   6.74e9, 32, 4096, 32, 32),   # MHA
    "Llama-3-8B":   Model("Llama-3-8B",   8.03e9, 32, 4096, 32,  8),   # GQA
    "Llama-2-13B":  Model("Llama-2-13B", 13.0e9,  40, 5120, 40, 40),
    "Qwen2.5-14B":  Model("Qwen2.5-14B", 14.7e9,  48, 5120, 40,  8),
    "Qwen2.5-32B":  Model("Qwen2.5-32B", 32.5e9,  64, 5120, 40,  8),
    "Llama-3-70B":  Model("Llama-3-70B", 70.6e9,  80, 8192, 64,  8),
    "Qwen2.5-72B":  Model("Qwen2.5-72B", 72.7e9,  80, 8192, 64,  8),
    "Mixtral-8x7B": Model("Mixtral-8x7B",46.7e9,  32, 4096, 32,  8,
                          active_params=12.9e9, n_experts=8, top_k=2, moe_layers=32),
    "DeepSeek-V3":  Model("DeepSeek-V3", 671e9,   61, 7168,128,128,
                          active_params=37e9, n_experts=256, top_k=8, moe_layers=58),
}


@dataclass
class Workload:
    label: str
    cls: str                       # "training" | "inference"
    flows: list[Flow] = field(default_factory=list)
    step_time_s: float = 0.0
    feasible: bool = True
    why_infeasible: str = ""
    meta: dict = field(default_factory=dict)

    # ── aggregates, all byte-weighted over the flows actually present ──
    def _bw(self, attr: str) -> float:
        tot = sum(f.bytes_per_s for f in self.flows)
        if tot <= 0:
            return 0.0
        return sum(f.bytes_per_s * getattr(f, attr) for f in self.flows) / tot

    @property
    def intra_bps(self) -> float:
        return sum(f.bytes_per_s for f in self.flows if f.fabric == "intra")
    @property
    def inter_bps(self) -> float:
        return sum(f.bytes_per_s for f in self.flows if f.fabric == "inter")
    @property
    def total_bps(self) -> float:
        return self.intra_bps + self.inter_bps
    @property
    def sym(self) -> float:          return self._bw("sym")
    @property
    def size_cv(self) -> float:      return self._bw("size_cv")
    @property
    def interval_cv(self) -> float:  return self._bw("interval_cv")
    @property
    def msg_bytes(self) -> float:
        return max((f.msg_bytes for f in self.flows), default=0.0)
    @property
    def collectives_per_step(self) -> float:
        return sum(f.count_per_step for f in self.flows)

    def as_row(self) -> dict:
        return dict(label=self.label, cls=self.cls, feasible=self.feasible,
                    intra_bps=self.intra_bps, inter_bps=self.inter_bps,
                    total_bps=self.total_bps, sym=self.sym, size_cv=self.size_cv,
                    interval_cv=self.interval_cv, msg_bytes=self.msg_bytes,
                    collectives_per_step=self.collectives_per_step,
                    step_time_s=self.step_time_s, **self.meta)


# ── memory feasibility ──────────────────────────────────────────────────────
#   Bytes per parameter held by one GPU under mixed-precision Adam:
#   2 (bf16 weight) + 2 (bf16 grad) + 4 (fp32 master) + 8 (fp32 m, v) = 16
BYTES_PER_PARAM_ADAM = 16
BYTES_PER_PARAM_SGD  = 8      # bf16 weight + bf16 grad + fp32 master

def _training_memory(model, dp, tp, pp, b_local, seq, b, sharding, checkpointing, opt):
    owned = model.params / (tp * pp) / (dp if sharding == "fsdp" else 1)
    per_param = BYTES_PER_PARAM_ADAM if opt == "adam" else BYTES_PER_PARAM_SGD
    state = owned * per_param
    layers_here = model.layers / pp
    # activation residency: ~2 tensors/layer with checkpointing, ~12 without
    act = b_local * seq * model.hidden * b * layers_here * (2 if checkpointing else 12)
    return state + act

def _inference_memory(model, tp, batch, prompt_len, output_len, b, kv_b):
    weights = model.params * b / tp            # all experts must be resident for MoE
    kv = batch * (prompt_len + output_len) * model.kv_bytes_per_token(kv_b) / tp
    return weights + kv


# ── training ────────────────────────────────────────────────────────────────
def training_comm(model: Model, hw: HW, *, dp=8, tp=1, pp=1, global_batch=256,
                  seq=2048, b=2, sharding="ddp", grad_accum=1, checkpointing=False,
                  optimizer="adam", mfu=0.45, diloco_inner=500) -> Workload:
    world = dp * tp * pp
    single_node = world <= hw.gpus_per_node
    dp_fabric = "intra" if single_node else "inter"
    b_local = max(1, global_batch // dp)
    layers_here = model.layers / pp
    act = b_local * seq * model.hidden * b        # activation crossing one TP collective
    p_shard = model.params / (tp * pp)            # params this GPU is responsible for

    # Two-pass: accumulate bytes PER STEP, then set the step time, then convert to
    # rates. A step cannot be shorter than the time its traffic needs on the wire,
    # so the step time is max(compute, intra/NVLink, inter/NIC) -- the standard
    # perfectly-overlapped model. Without this the grid contains configurations
    # moving more bytes per second than the fabric can carry.
    pending = []
    def add(kind, total_bytes, msg, count, fabric, icv=STEP_CV):
        if total_bytes > 0:
            pending.append((kind, total_bytes, msg, count, fabric, icv))

    if tp > 1:      # 2 all-reduces forward + 2 backward, per layer on this stage
        add("allreduce", 4 * layers_here * f_allreduce(tp) * act,
            act, 4 * layers_here, "intra")
    if pp > 1:      # activations handed to the next stage, gradients back
        add("p2p", 2 * act, act, 2, "intra" if single_node else "inter")
    if dp > 1:
        if sharding == "fsdp":
            ag = 3 if checkpointing else 2        # fwd + bwd (+ recompute) all-gathers
            add("allgather", ag * f_allgather(dp) * p_shard * b, p_shard*b, ag, dp_fabric)
            add("reducescatter", f_reducescatter(dp) * p_shard * b, p_shard*b, 1, dp_fabric)
        elif sharding == "diloco":                # one outer sync per H inner steps
            add("allreduce", f_allreduce(dp) * p_shard * b / diloco_inner,
                p_shard*b, 1/diloco_inner, dp_fabric)
        else:                                     # ddp
            add("allreduce", f_allreduce(dp) * p_shard * b / grad_accum,
                p_shard*b, 1/grad_accum, dp_fabric)
    if model.n_experts and model.moe_layers:      # dispatch + combine per MoE layer
        moe_here = model.moe_layers / pp
        add("alltoall",
            2 * moe_here * f_alltoall(dp) * b_local * seq * model.top_k * model.hidden * b,
            b_local * seq * model.top_k * model.hidden * b / dp, 2 * moe_here, dp_fabric)

    flops = (8 if checkpointing else 6) * model.active * global_batch * seq * grad_accum
    t_compute = flops / (world * hw.flops_bf16 * mfu)
    b_intra = sum(x[1] for x in pending if x[4] == "intra")
    b_inter = sum(x[1] for x in pending if x[4] == "inter")
    step_time = max(t_compute, b_intra / hw.nvlink_bw, b_inter / hw.nic_bw)
    comm_bound = step_time > t_compute * 1.001
    flows = [Flow(k, tb / step_time, msg, cnt, fab, icv)
             for k, tb, msg, cnt, fab, icv in pending]

    mem = _training_memory(model, dp, tp, pp, b_local, seq, b, sharding,
                           checkpointing, optimizer)
    ok = mem < 0.85 * hw.hbm_bytes
    return Workload(
        label=f"{model.name}/{sharding}/dp{dp}tp{tp}pp{pp}", cls="training",
        flows=flows, step_time_s=step_time, feasible=ok,
        why_infeasible="" if ok else f"needs {mem/1e9:.0f} GB of {hw.hbm_bytes/1e9:.0f}",
        meta=dict(model=model.name, hw=hw.name, dp=dp, tp=tp, pp=pp, world=world,
                  sharding=sharding, global_batch=global_batch, seq=seq,
                  grad_accum=grad_accum, checkpointing=checkpointing,
                  mem_gb=mem/1e9, single_node=single_node, comm_bound=comm_bound,
                  compute_time_s=t_compute, mfu_effective=mfu*t_compute/step_time))


# ── inference ───────────────────────────────────────────────────────────────
def inference_comm(model: Model, hw: HW, *, mode="disaggregated", tp=8, pp=1,
                   replicas=1, batch=64, prompt_len=2048, output_len=256, b=2,
                   kv_b=2, mbu=0.7) -> Workload:
    """mode: 'colocated'     prefill and decode on the same GPUs; TP traffic only
             'disaggregated' KV cache pushed prefill -> decode across the NIC
             'replicated'    one full model per GPU; no interconnect traffic at all
    """
    gpus_per_replica = tp * pp
    gpus = gpus_per_replica * replicas
    spans_nodes = gpus_per_replica > hw.gpus_per_node
    pp_fab  = "inter" if spans_nodes else "intra"
    moe_fab = "inter" if spans_nodes else "intra"
    # decode is memory-bandwidth bound: the weights stream once per decode step
    decode_step_s = (model.active * b) / (gpus_per_replica * hw.hbm_bw * mbu)
    tokens_per_s = batch * replicas / decode_step_s
    qps = tokens_per_s / output_len
    kv_tok = model.kv_bytes_per_token(kv_b)

    pending = []
    def add(kind, bytes_per_step, msg, count, fabric, icv):
        if bytes_per_step > 0:
            pending.append((kind, bytes_per_step, msg, count, fabric, icv))

    if mode != "replicated" and tp > 1:
        # decode emits one token per sequence per step, so the tensor is tiny
        act_dec = batch * model.hidden * b
        if pp > 1:      # stage -> stage activations, directional, over the NIC
            add("p2p", 2 * act_dec, act_dec, 2, pp_fab, STEP_CV)
        add("allreduce", 4*(model.layers/pp)*f_allreduce(tp)*act_dec,
            act_dec, 4*model.layers/pp, "intra", STEP_CV)  # decode is metronomic too
        if model.n_experts and model.moe_layers:
            add("alltoall",
                2*(model.moe_layers/pp)*f_alltoall(gpus_per_replica)
                *batch*model.top_k*model.hidden*b,
                batch*model.top_k*model.hidden*b, 2*model.moe_layers, moe_fab, STEP_CV)
    if mode == "disaggregated":
        add("kv_push", qps * prompt_len * kv_tok / gpus * decode_step_s,
            prompt_len * kv_tok, 1, "inter", REQUEST_CV)  # request arrivals, Poisson

    b_intra = sum(x[1] for x in pending if x[4] == "intra")
    b_inter = sum(x[1] for x in pending if x[4] == "inter")
    t_mem = decode_step_s
    decode_step_s = max(decode_step_s, b_intra / hw.nvlink_bw, b_inter / hw.nic_bw)
    comm_bound = decode_step_s > t_mem * 1.001
    tokens_per_s = batch * replicas / decode_step_s
    qps = tokens_per_s / output_len
    flows = [Flow(k, tb / decode_step_s, msg, cnt, fab, icv)
             for k, tb, msg, cnt, fab, icv in pending]

    mem = _inference_memory(model, gpus_per_replica, batch, prompt_len, output_len, b, kv_b)
    ok = mem < 0.90 * hw.hbm_bytes
    return Workload(
        label=f"{model.name}/infer-{mode}/tp{tp}pp{pp}", cls="inference", flows=flows,
        step_time_s=decode_step_s, feasible=ok,
        why_infeasible="" if ok else f"needs {mem/1e9:.0f} GB of {hw.hbm_bytes/1e9:.0f}",
        meta=dict(model=model.name, hw=hw.name, mode=mode, tp=tp, pp=pp, replicas=replicas,
                  batch=batch, prompt_len=prompt_len, output_len=output_len,
                  qps=qps, gpus=gpus, mem_gb=mem/1e9, world=gpus, seq=prompt_len,
                  single_node=not spans_nodes, comm_bound=comm_bound,
                  compute_time_s=t_mem))
