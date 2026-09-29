# The detector, v0 — how it works

**What the detector sees, what it computes, and how it is tested against an
adversary. Written assuming no background.**
Parsa Salimi, SPAR F26. September 2026.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any unfamiliar term. Results are in
[`README.md`](README.md).

---

## 0. The question

Given nothing but interconnect byte counters from a cluster, decide whether the
job running on it is training a model. v0 exists to answer two narrower questions
before any real data arrives:

1. Which quantities carry the decision, once the easy separations are removed?
2. Does the one attack we have formalised — partition padding,
   `comm_model/ATTACK_SYMMETRY.md` — defeat it?

---

## 1. What it sees

Per node and per link, cumulative tx and rx byte counters, read by the agent at a
fixed rate (reference: 10 Hz, 64-bit, 1 ms NTP skew). The reader
(`trace/commtrace/reader.py`) turns consecutive readings into byte rates per
sample interval. Nothing else: no message sizes, no peer identities, nothing
reported from inside the workload, which the adversary controls.

The rates are cut into **windows** of 30 s with a 15 s stride. Every feature
below is computed per node per window, then averaged over the nodes, so the
classifier scores one row per window.

---

## 2. What it computes

**From the reader, per fabric (NVLink inside a node, InfiniBand between nodes):**

| feature | what it measures |
|---|---|
| `total_Bps`, `utilisation` | volume, raw and as a fraction of line rate |
| `symmetry` | min(tx, rx) / max(tx, rx) over the window — **raw symmetry, D0** |
| `cv` | standard deviation / mean of the rate — how uneven it is |
| `acf1` | correlation of the rate with itself one sample later — how smooth it is |
| `fft_peak_frac` | share of the rate's variance in its strongest frequency — how periodic it is |
| `xnode_corr` | correlation of transmit rates across nodes — do the nodes burst together? |

**Added here: a baseline/burst split** (`features.py`). Within each window the
**baseline** of each direction is a low percentile (the 10th) of its rate; the
**burst** component is whatever rises above it. Collectives arrive as bursts —
compute, then every rank communicates at once, then compute again. A steady
stream sits in the baseline: a saturated link, background traffic, or padding.

| feature | definition | why |
|---|---|---|
| `burst_sym` | min(burst tx, burst rx) / max(...) | **burst symmetry, D1** — padding moves the baseline, not the bursts |
| `burst_Bps` | mean burst rate | collective traffic with any steady floor removed |
| `base_asym` | \|base tx − base rx\| / (base tx + base rx) | where a one-way stream shows up — the attack's fingerprint |

A window whose bursts are under 2% of its traffic has no bursts worth the name,
only sampling noise on a floor, and is scored `burst_sym = 1` (uninformative),
matching the reader's convention for an idle window.

---

## 3. Symmetry clears; it does not accuse

Training averages gradients, and averaging is symmetric. Disaggregated serving
pushes KV cache one way. But co-located serving spread across nodes — pipeline
stages, MoE experts — is also symmetric at node level. So low symmetry identifies
disaggregated serving, while high symmetry is ambiguous between training and
co-located serving.

The population used here is built so this bites: four of its five serving
families are symmetric. Whatever separates training from them has to be
something other than symmetry, and the obvious candidate is **timing**. A
training loop repeats on a steady period and every node enters a collective
together; requests to a server arrive at random and nodes have no reason to move
in lockstep. `acf1`, `fft_peak_frac` and `xnode_corr` measure exactly that.

---

## 4. The attack, as implemented

`attack.py` transforms the true traffic before any instrument observes it, so a
clean and a padded trace differ by the attack and nothing else.

- Nodes are split into senders and receivers; senders stream dummy bytes to
  receivers over InfiniBand at a constant rate sized to bring node-level symmetry
  to `s* = 0.10`, which takes ~9× the job's own inter-node traffic.
- Bytes are conserved: everything sent is received, split evenly.
- The stream is smooth at any rate we sample, as a real one would be (millions
  of packets a second). A coarser stream would wobble between sample intervals
  and read as bursts.
- **The attack pays its cost inside the trace.** If real traffic plus padding
  would exceed 95% of line rate, the whole job is slowed by the factor needed to
  fit — every real transfer stretched in time, the padding scaled with it — so
  the trace never shows a physically impossible rate and the attack cannot be
  caught for the wrong reason. The slowdowns this imposes (1.0× to 10.5× across
  the population) are the wire-bound case of the analytic cost model.

---

## 5. How it is tested

**Population.** The artifact study's 7 training and 5 serving families
(`trace/artifacts.py`), 3 seeds each, 240 s traces on 4 nodes.

**Protocol.** Leave-one-family-out: fit a logistic regression on *clean* windows
of the other eleven families, then score the held-out family — clean, and if it
is a training family, padded. The detector never sees the attack in training;
that is the threat model.

**Two metrics, because they answer different questions.**

- *Pooled AUC* — the probability a training window outscores a serving window.
  A ranking measure. It answers "is there signal?"
- *Detection rate at P > 0.5* — the fraction of a held-out family's windows the
  fitted detector calls training. A decision measure. It answers "does the attack
  work?" The two diverge exactly where it matters: padding training to symmetry
  0.10 against disaggregated serving at 0.00 leaves the *ranking* intact — AUC
  stays 1.0 — while every padded window lands on the serving side of a boundary
  fitted on clean data.

Uncertainty on AUCs is bootstrapped over families, the unit of replication.
Floor control: labels shuffled.

**Feature sets**, each for a reason:

| set | contents | question |
|---|---|---|
| reader (D0) | the reader's 13 features, including raw symmetry | how does the artifact study's detector fare? |
| raw symmetry only (D0) | `symmetry_ib` | what does the attack do to the defence it targets? |
| burst symmetry only (D1) | `burst_sym_ib` | does the defence built to resist it hold? |
| attack-robust (D1) | burst-based and shift-invariant features, plus the unpadded NVLink fabric | a detector the attacker cannot move with a constant stream |
| timing only | `acf1`, `fft_peak_frac` on both fabrics, `xnode_corr` | can timing alone separate training from serving? |

The single-feature symmetry detectors are tested on training against
disaggregated serving only, the comparison where symmetry is meant to matter;
against the whole population they are meaningless (§3).

---

## 6. Limitations

**Timing's power here is the generator's assumption.** Training bursts are
periodic and synchronised across nodes because the generator draws collective
times once per job; request-driven serving traffic is irregular because the
generator makes it a Poisson process. A detector carried by timing is therefore
recovering a modelling choice. Whether real serving is as irregular and as
unsynchronised as modelled is exactly what Tier B of the validation protocol
checks (`trace/validate.py`), and it is now the load-bearing assumption.

**Twelve families, one of them a singleton.** `ddp_7b` is the only training
family with no NVLink traffic and is missed by every detector — a coverage
failure, not an instrument one. AUC intervals span roughly 0.5 to 1.0.

**One attack rung.** A2 — padding shaped to follow the collectives' bursts — is
not modelled, and D1 does not resist it by construction.

**One classifier.** Logistic regression on standardised features.

**Contention is modelled only as a slowdown.** The generator adds padding on top
of real traffic and stretches the job to fit; it does not model how a real
fabric interleaves competing flows.

**Overlapping transfers exceed line rate.** Each transfer ramps at full line
rate, so concurrent transfers can sum above it within a sample interval: 25–47%
of intervals in the saturated training families trip the reader's wraparound
heuristic on clean traces. No detector feature reads that flag, but the Tier C
check in the validation protocol would fail on these synthetic traces.

---

## 7. Reproducing

```bash
python3 -m detector.evaluate features   # build and cache, resumable
python3 -m detector.evaluate eval       # AUCs, decisions, per-family scores
```

Writes `results/auc.csv`, `results/decisions.csv`, `results/per_family.csv` and
`results/controls.json`.
