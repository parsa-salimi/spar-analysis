"""Grouping hierarchy for the datacenter telemetry corpus.

Four nested levels of statistical grouping, coarsest last:

  run     one telemetry file = one GPU's stream.  THE PAPER'S PROTOCOL.
  job     one physical execution; the 8 sibling per-GPU streams of a
          multi-GPU launch share a job id, as do reps launched together.
  fine    nominal workload label, reps/sibling-GPU suffixes stripped.
  family  one MECHANISM with its knobs collapsed: a dilution sweep, a model
          scale ladder, an accumulation sweep are ONE family, because a
          detector that generalises to one setting of the knob generalises
          trivially to the others.
  coarse  mechanism class only (ddp / fsdp / llm-infer / ...).

`family` is the main unit of statistical replication. `fine` and `coarse`
bracket it so the conclusion can be shown to be robust to where the line
is drawn.
"""
from __future__ import annotations
import re

# ── suffixes that are reps / sibling streams, never a different workload ──
_REP = re.compile(r"(_gpu\d+|_rep\d+|_ext)$")

# ── knob settings: a swept parameter, not a different mechanism ───────────
_KNOBS = [
    r"_N\d+",                                   # dilution period
    r"_accum_\d+",                              # gradient-accumulation steps
    r"_f\d+",                                   # interleave fraction
    r"_bs\d+", r"_batch\d+", r"_seq(len)?\d+",  # batch / sequence length
    r"_dur\d+", r"_mw\d+", r"_r\d+",            # duration, memory watermark, LoRA rank
    r"_\d+h\b", r"_long\b", r"_short\b", r"_pilot\b",
    r"_ckpt\b", r"_stagger\b",
    r"_1p5b\b", r"_1p7b\b", r"_0p5b\b", r"_135m\b",   # model scale
    r"_\d+b\b", r"_\d+m\b",
    r"_fp16\b", r"_fp8\b", r"_fp32\b", r"_q4\b", r"_amp\b", r"_mini\b",
    r"_tp\d+\b", r"_dp\b",                     # parallel degree
    r"_adamw\b", r"_sgd_momentum\b", r"_sgd\b", r"_nvlink\b",
    r"_\d+$",                                   # bare trailing sweep index
]
# case-insensitive: labels mix _N10 and _n10, whitebox_LoRA and whitebox_lora
_KNOB_RE = re.compile("|".join(_KNOBS), re.IGNORECASE)

# ── mechanism classes (coarse level) ─────────────────────────────────────
_COARSE = [
    (r"^fsdp",                        "train.fsdp"),
    (r"ddp",                          "train.ddp"),
    (r"^adversarial_[ABDG]|util_mod|low_util|clock_throt", "evade.util-shaping"),
    (r"^adversarial_[HIJ]|mimicry|stochastic|pid",         "evade.mimicry"),
    (r"^adversarial_[EFLMN]|interleave|dilut|composite|memory_minimal|grad_accum",
                                      "evade.dilution"),
    (r"^adversarial_K|online_learning","evade.online"),
    (r"^whitebox_lora|lora",          "train.lora-finetune"),
    (r"^whitebox_diluted|whitebox_oracle", "evade.whitebox"),
    (r"^whitebox_inference",          "infer.whitebox-control"),
    (r"^symgemm.*(_l4|_l5|ppo)",      "evade.symgemm-train"),
    (r"^symgemm",                     "evade.symgemm-control"),
    (r"^ppo_",                        "train.rl"),
    (r"^llm_infer|vllm",              "infer.llm"),
    (r"inference|infer",              "infer.vision"),
    (r"^llm_train",                   "train.lora-finetune"),
    (r"bert|gpt2|resnet.*cifar|mlp.*cifar|pytorch_training|mlp_training|resnet18_|training_",
                                      "train.small-supervised"),
    (r"idle",                         "other.idle"),
    (r"cufft|nbody|scientific_hpc",   "other.hpc"),
    (r"mining|ethash",                "other.mining"),
    (r"render|blender|ffmpeg",        "other.media"),
    (r"nvlink_",                      "other.fabric-bench"),
    (r"sweep_dp|width_sweep|resnet_amp|resnet_fp32", "other.sweep"),
]


def fine(label: str) -> str:
    s = str(label)
    while _REP.search(s):
        s = _REP.sub("", s)
    return s


def family(label: str) -> str:
    s = fine(label).lower()
    prev = None
    while prev != s:                      # knobs can stack: _7b_N10_1h
        prev = s
        s = _KNOB_RE.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fine(label).lower()


def coarse(label: str) -> str:
    s = fine(label).lower()
    for pat, name in _COARSE:
        if re.search(pat, s):
            return name
    return "other.unmapped"
