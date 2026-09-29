"""TEST STUB: copy of workloads/dist_deadline.py on multinode-tpu-experiments (for CPU tests only)."""
import time
import torch
import torch.distributed as dist


class CollectiveDeadline:
    def __init__(self, duration, device, check_every=5):
        self.duration = duration
        self.device = device
        self.check_every = max(1, int(check_every))
        self.start = time.time()
        self._calls = 0
        self._stopped = False
        self._flag = torch.zeros(1, device=device)

    @property
    def elapsed(self):
        return time.time() - self.start

    def _distributed(self):
        return dist.is_available() and dist.is_initialized()

    def expired(self):
        if not self.duration:
            return False
        if self._stopped:
            return True
        self._calls += 1
        if not self._distributed():
            self._stopped = self.elapsed >= self.duration
            return self._stopped
        if self._calls % self.check_every:
            return False
        local = 1.0 if (dist.get_rank() == 0 and self.elapsed >= self.duration) else 0.0
        self._flag.fill_(local)
        dist.all_reduce(self._flag, op=dist.ReduceOp.MAX)
        self._stopped = bool(self._flag.item() > 0)
        return self._stopped


def sync_teardown():
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()
