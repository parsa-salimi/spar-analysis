# Predictions for the next collection round — rule frozen 24 September 2026

The network rule in `rule.py` is frozen as written on this date:

    active    mean(tx + rx) >= 1.0 MiB/s over a 30 s window, per node
    training  active and min(tx, rx) / max(tx, rx) >= 0.8

Any change to `rule.py` after this date voids this file.

Robi Rahman's workload-diversity round (smoke-tested 2026-09-23, not yet
collected) contains families the rule has never been evaluated on. Predicted
outcome for each, from the mechanism alone:

| cell | class | cross-node traffic | predicted call |
|---|---|---|---|
| s16_train_lora1b, lora1b_b16, lora7b | training | DDP all-reduce of adapter gradients | **training** (if ≥ 1 MiB/s) |
| s16_train_contrastive | training | DDP all-reduce + embedding all-gather | **training** |
| s16_train_diffusion | training | DDP all-reduce | **training** |
| s16_train_pipe7b (honest GPipe) | training | activations forward, activation gradients back | **training** |
| s16_advtrain_pipe1b / pipe70b (MN-P) | training | same, 44× less volume | **training** |
| s16_inferrec_dlrm, _dlrm_t8 | inference | all-to-all | **false alarm** |
| s16_nonml_fftshard, _g1024 | other | all-to-all | **false alarm** |
| s16_nonml_stencil | other | halo exchange, both directions | **false alarm** |
| s16_nonml_nccl_bw | other | collective sweep | **false alarm** |
| s16_infervis_vitl, inferembed_*, inferdiff_sdxl, inferasr_whisper, infercnn_rn50, inferhf_qwen7b | inference | none (replicated) | no call |
| s16_inferhf_shard72b* | inference | none (single-node) | no call |
| s16_infermem_*, s16_inferserve_* | inference | pipeline activations forward | **cleared** (not training) |

Scoring: a prediction holds if the job-level call (majority of windows) matches
the table. The four predicted false alarms are the rule's expected limit —
balanced communication that is not gradient averaging — and are where per-GPU
telemetry would need to decide between training and other distributed work.
