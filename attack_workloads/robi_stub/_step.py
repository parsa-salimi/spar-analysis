"""TEST STUB: copy of workloads/_step.py on multinode-tpu-experiments (for CPU tests only)."""
import time


def emit_step(kind, batch_idx, *, tokens=0, passes=1, node=0, rank=0, **kv):
    line = (f"STEP ts={time.time():.3f} kind={kind} batch_idx={batch_idx} "
            f"tokens={tokens} passes={passes} node={node} rank={rank}")
    for k, v in kv.items():
        line += f" {k}={v}"
    print(line, flush=True)
