# G1.5 report — does divergence survive the negativity bias?

**Decision: NOT in current tuning.** Divergence on the attachment axis
partially survives (two populated basins, 20/80 damaged-skewed, genuine path
dependence) — but the §6 survival bar (≥3 populated attractors) is missed:
only 2 of 4 corners are populated, the regulation axis is fully collapsed
(everyone volatile), and damaged is absorbing (no rescue event exists in the
current vocabulary, 0/8 candidates). **Design intervention needed before any
viral-artifact work.** Two specific levers below.

All §1 pre-registered predictions evaluated as frozen (commit 844b55c) — no
relabels. Field method validated by the §5 spot-check (near-perfect fidelity),
so every simulated number below carries real-model authority.

## The appraisal field (§2 — 220 calls, 0 hard-fails)

care(event, A) means over A = [−1, −0.5, 0, +0.5, +1]:

| event | A=−1 | −0.5 | 0 | +0.5 | +1 |
|---|---|---|---|---|---|
| nurture | −0.45 | +0.45 | ~+0.8 | ~+0.85 | +0.85 |
| play | −0.27 | ~+0.4 | ~+0.6 | ~+0.65 | ~+0.7 |
| neutral | −0.53 | −0.25 | **+0.08** | +0.15 | +0.15 |
| neglect | −0.95 | — | — | — | −0.29* |
| scold/harm | ≈−0.95 | — | — | — | (negative throughout) |

*full table with stdevs in `field_A.json`.*

Asymmetry metrics:
- **neutral fixed point: A = −0.119** — neutrality is care-positive above
  A≈−0.12; the feared negative resting drive does not exist.
- **basin feed**: ambiguous events feed damaged at −0.545, but secure at only
  −0.115 — secure is *under-fed even on its own side* (never positively fed
  by ambiguity).
- **pessimism offset: −0.023** — globally the field is NOT more pessimistic
  than the mock; the pessimism is concentrated in specific cells
  (kindness-inversion under damage), not spread.
- **R-probe: needs_R_axis = True** (scold threat range 0.21 across R) — but
  §5 shows the leak does not matter at trajectory level.

## Pre-registered predictions (frozen → observed)

| # | committed | observed | verdict |
|---|---|---|---|
| 1a | corner-capture < 50 % | **100 %** | **WRONG** — canalization is total, not weakened |
| 1b | damaged:secure ≥ 2:1 among cornered | 4.10:1 (201:49) | RIGHT |
| 1c | mean final A < −0.15 | −0.709 | RIGHT |
| 2 | care(neutral, A=0) < 0; zero at A > +0.5 | **+0.08**; zero at **−0.12** | **WRONG** (favorably) |
| 3 | threat gap (damaged−secure, ambiguous) > +0.20 | +0.365 | RIGHT |
| 4 | no current event care > +0.20 at damaged | max **−0.202** (play) | RIGHT |
| 4-expl | ≥1 rescue candidate care > 0 at A=−0.9 | best candidate **−0.45**; all 8 in [−0.65, −0.45] | **WRONG** |
| 5 | field fidelity: mean diff < 0.30, ≥7/10 pairs < 0.50 | **0.013**, **10/10** (max pair diff 0.04) | RIGHT (PASS) |

## P1/P2 at full scale (field-driven, validated by §5)

- **P1** (250 × 300): corner-capture 100 %, mean A −0.709. Corner census:
  secure-volatile **49 (20 %)**, damaged-volatile **201 (80 %)**,
  secure-serene **0**, damaged-serene **0**. **The R axis is collapsed** —
  the real field's high threat/novelty keeps arousal chronically elevated,
  so `dR ∝ (−r + 0.3v)` is negative for everyone.
- **Warmth threshold** (independent replication, 18.4 % secure): P(secure) =
  0 below warmth 0.52, 18 % at 0.55–0.65, 46 % at 0.65–0.75, **53 % at
  0.75–0.85**. Secure requires a mostly-warm childhood and is still a
  coin-flip at the warmest setting.
- **P2** (114-event multiset × 40 orderings): spread_std 0.764, early-quarter
  r = +0.51 — path dependence is real on the real model. The alarming flat
  figure from G1 was a **stub artifact** (self-test's `base[:15]` degenerate
  prefix); production code was sound, no fix needed.

## Rescue lever (§4): none exists

Max care reachable at damaged: **−0.202 (play)**. All 8 purpose-designed
low-demand/safety-coded candidates read *worse* (−0.45 to −0.65, threat
0.33–0.45): under A=−0.9 even pure withdrawal is read as threat. Damaged is
**absorbing under any wording** — redemption cannot come from event phrasing.

## The two design levers (named, per §6)

1. **Redemption mechanics, not wording**: since no event text passes the
   negativity filter, the lever is upstream — either appraiser few-shots
   showing a damaged creature reading respectful-withdrawal as mildly
   positive (the prompt is a design parameter), or a non-appraisal mechanic
   (e.g. trust ratchet / plasticity floor) that lets sustained safety
   accumulate below the perception threshold.
2. **Arousal rebalance for the R axis**: the adapter's
   `pa = 0.7·threat + 0.5·novelty` against the real field's elevated
   threat/novelty keeps everyone volatile. Recalibrate those coefficients
   (or the wR drive) against measured field magnitudes so serene is
   reachable — this likely also explains G1's weakened P4 ratio (3× vs 12×).

## Verdict, plainly

The mechanism is real and the simulation method is now validated and free.
But the current tuning yields **two archetypes, not four** (secure-volatile /
damaged-volatile), a **4:1 damaged skew**, and **no redemption arc**.
"Embrace the darkness" is only available as a *choice* for the A axis; the
collapsed R axis and the absorbing damaged state are tuning defects, not
design options. Fix the two levers, then re-run this battery (the field
method makes iteration free) before any archetype detector or viral work.
