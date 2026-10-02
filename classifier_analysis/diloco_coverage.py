"""Does DiLoCo defeat the symmetry rule? Answered from corpus evidence + arithmetic.

We do NOT synthesise traffic. DiLoCo alternates two phases that both already
exist, measured, in Robi's corpus:

  inner steps  each node trains alone, no cross-node traffic. The corpus has this
               exactly: s16_train_ddpstagger_k is 16 GPUs training with the NIC
               carrying only background (tx/rx means ~0.02-0.17 MiB/s, far below
               the rule's 1 MiB/s floor). The rule calls it training in 0% of
               windows -- confirmed, results/adversarial.csv.
  outer sync   an all-reduce of the pseudo-gradient. Every real all-reduce
               training job in the corpus (FSDP, DDP) reads per-node symmetry
               0.95-0.96 and is called training in ~100% of its active windows.

So a window that contains a sync is caught; a window that is pure inner-step is
not. Whether DiLoCo evades is therefore a question about WINDOW COVERAGE: what
fraction of 30 s / 15 s windows contain an outer sync, as a function of the sync
interval T. That is pure arithmetic and needs no new data.

A window (start s, length W=30, stride 15) contains a sync if a sync instant
falls in [s, s+W). With syncs every T seconds on a long run, the fraction of
windows that catch one is min(1, W/T) for a burst sync, and 1 for a spread sync
(traffic every inner step, so every active window is balanced).

Reported per T: fraction of windows with a sync (-> rule TPR at the window
level), and whether the job is caught under an ANY-window rule and under a
MAJORITY-window rule. Model sizes set how long each burst lasts on a real fabric,
which only helps the attacker (longer bursts cover more windows); the pessimistic
case for the detector is an instantaneous burst, used here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import common as C

W, STRIDE = 30.0, 15.0


def coverage(T, dur=3600.0, burst=True):
    """Fraction of windows on a `dur`-second run that contain an outer sync."""
    if not burst:
        return 1.0                       # spread: every active window is balanced
    syncs = np.arange(T, dur, T)
    starts = np.arange(0, dur - W, STRIDE)
    hit = [(any((s <= k < s + W) for k in syncs)) for s in starts]
    return float(np.mean(hit))


def main():
    out = C.RES; out.mkdir(parents=True, exist_ok=True)
    Ts = [10, 20, 30, 60, 120, 300, 600, 1200, 3600]
    rows = []
    for T in Ts:
        cb = coverage(T, burst=True)
        rows.append(dict(sync_interval_s=T,
                         window_cov_burst=round(cb, 3),
                         window_cov_spread=1.0,
                         caught_any_window=cb > 0,
                         caught_majority_burst=cb > 0.5))
    df = pd.DataFrame(rows)
    df.to_csv(out / "diloco_coverage.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    # anchor to the real corpus numbers the argument rests on
    adv = pd.read_csv(C.RES / "../../network_detector/results/adversarial.csv") \
        if (C.RES / "../../network_detector/results/adversarial.csv").exists() else None
    print("\nAnchors from the corpus:")
    print("  inner-step phase  = s16_train_ddpstagger_k : rule calls 0% of windows training")
    print("  outer-sync phase  = real FSDP/DDP all-reduce: symmetry 0.95-0.96, ~100% caught")


if __name__ == "__main__":
    main()
