"""A network detector that is derived, not learned.

    active    mean(tx + rx) >= ACTIVE_MIBPS in the window
    training  active and node-level symmetry >= BALANCED

Why a rule: a classifier fitted on seven workload families cannot learn that
balanced traffic means training when, with a family held out, the training set
contains no unbalanced negative -- it learns "traffic present means training"
instead. The rule encodes the mechanism directly: gradient averaging is an
all-reduce, and an all-reduce sends as much as it receives, whatever the model.
It needs no coverage because it is not fitted.

Provenance of the thresholds, stated plainly: the structure comes from the comm
model (comm_model/), the thresholds were chosen after the per-cell medians of the
existing corpus had been seen. On this corpus the rule is therefore a
description, not a held-out test. The held-out test is the next collection round,
against which the rule is frozen as written here.
  ACTIVE_MIBPS = 1.0    ~50x the idle background (0.01-0.03 MiB/s), ~10x below the
                        quietest real cross-node workload
  BALANCED     = 0.8    all-reduce reads 0.954-0.961 on this NIC (received bytes
                        carry TCP/IP headers, sent bytes do not); pipeline serving
                        reads 0.04-0.18
"""
ACTIVE_MIBPS = 1.0
BALANCED = 0.8


def call(net_windows):
    import numpy as np
    mean_rate = 10 ** net_windows.vol - 1
    active = mean_rate >= ACTIVE_MIBPS
    return active, active & (net_windows.sym >= BALANCED)
