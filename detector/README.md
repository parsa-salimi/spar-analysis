# Detector v0 — results

Read [`METHOD.md`](METHOD.md) first. All numbers are on synthetic traces from the
generator, 7 training and 5 serving families, held out one family at a time, the
detector fitted on clean data only.

---

## 1. The attack against the defence it targets

Training against disaggregated serving, one-feature detectors, fraction of
windows called training:

| defence | clean | padded to s* = 0.10 |
|---|---:|---:|
| raw symmetry (D0) | 100% | **0%** |
| burst symmetry (D1) | 100% | **100%** |

Padding completely defeats raw symmetry and does not touch burst symmetry. A1
costs the adversary a median 4.26× slowdown across the configuration grid and
buys nothing against D1.

## 2. The full detectors

Fraction of windows called training, mean over held-out families:

| detector | training, clean | training, padded | serving (false positives) |
|---|---:|---:|---:|
| reader (D0) | 69% | 74% | **40%** |
| attack-robust (D1) | 74% | 74% | 2.4% |
| timing only | 81% | 81% | **0%** |

| detector | AUC clean [95% CI] | AUC attacked | AUC vs symmetric serving only |
|---|---:|---:|---:|
| reader (D0) | 0.797 [0.48, 1.00] | 0.757 | 0.747 |
| attack-robust (D1) | 0.839 [0.54, 1.00] | 0.839 | 0.798 |
| timing only | 0.806 [0.50, 1.00] | 0.806 | 0.806 |

Floor, labels shuffled: 0.521.

**What this says.**

- **On this population detection is carried by timing, not symmetry.** Five
  timing features alone detect 81% of training windows with no false positives,
  and nothing the attack does moves them. The attack defeats a defence the full
  detector was barely using.
- **The reader's raw-volume features cause false positives.** The artifact
  study's detector calls two of five serving families training — co-located
  70B serving and MoE serving, both heavy symmetric traffic. Restricting to
  burst-based and shift-invariant features cuts that from 40% to 2.4%.
- **Symmetry clears; it does not accuse.** Against the four symmetric serving
  families, symmetry alone scores at or below chance, as it must.

## 3. Where it fails

| family | detected by timing | why |
|---|---:|---|
| `ddp_7b` | 0% | the only training family with no NVLink traffic; nothing covers it |
| `diloco_13b` | 64% | syncs once per 100 steps; most windows contain no collective at all |

The first is the power analysis's coverage failure in miniature. The second is
the evasion that looked free in the comm model looking partly free here too.

## 4. What it does not establish

That timing works on real hardware. The generator makes training periodic and
synchronised and serving irregular by construction, so a timing-based detector is
recovering the generator's assumptions. The first real traces have to check
those assumptions before this detector means anything — Tier B of
`trace/validate.py`.

## 5. Next

- **Validate timing first.** Pre-register the Tier B tolerances (burst duty
  cycle, interval regularity, cross-node correlation) before any real trace.
- **A2.** Padding shaped to follow the collectives. D1 does not resist it; timing
  might.
- **A second tensor-parallel-degree-1 training family**, so no mechanism in the
  population is a singleton.
- **Volume anomaly.** Padding moves ~10× the bytes the claimed serving workload
  needs; a check against the workload's plausible volume is untested.
