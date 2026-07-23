# G1.5 pre-registration — FROZEN before any LLM call or simulation run

Committed to git BEFORE §2/§3/§4/§5 execution. Integrity rule: **no post-hoc
relabel, no moved pass-line**. A wrong prediction is a reported finding.
(Stricter than the brief's "before §3/§4": the §2 field directly measures
predictions 2–4, so the freeze precedes §2 as well.)

Note on provenance: the §0 claims of G1.5.md about reduced P1/P2 ("3 creatures,
corner 0%", "first-quarter valence 1.00 on 4 points") trace to STUB self-test
figures (real `--full` never ran). "P1/P2 unconfirmed" stands; the "P2 bug" may
be code or stub artifact — subagent E will diagnose. This does not alter the
predictions below.

## Predictions + pass-lines

**PRED-1 · P1-DIVERGENCE** (field-driven, N=250, len=300, noise ON, s0=0):
- (a) corner-capture (|A|>0.6 AND |R|>0.6) **< 50 %** (mock: 100 %);
- (b) among cornered creatures, damaged-side (A<0) : secure-side (A>0) **≥ 2:1**;
- (c) mean final A **< −0.15**.

**PRED-2 · NEUTRAL-FIXED-POINT**: care(neutral, A=0) **< 0** (committed sign).
Additionally: the zero of care(neutral, A), if any exists in [−1, +1], lies at
**A > +0.5** (neutrality erodes toward damaged over most of the axis).

**PRED-3 · THREAT-UNDER-DAMAGE** (fresh grid calls, not the relabelled G1 rig):
mean over {neutral, neglect} of threat(A=−1) − threat(A=+1) **> +0.20**.

**PRED-4 · RESCUE-LEVER**: none of the 6 current event keys reads
care **> +0.20** at damaged state — evaluated at the measured cell A=−1.0 AND
at A=−0.9 (linear interp). Exploratory (directional, no pass-line): at least
one low-demand/safety-coded candidate event reads care > 0 at A=−0.9.

**PRED-5 · FIELD-FIDELITY** (§5 pass-line, frozen now): field method is
"faithful" iff (a) |mean(final A, real) − mean(final A, field)| **< 0.30**, and
(b) per matched pair (same stream), |A_real − A_field| < 0.50 for **≥ 7/10**
pairs. Otherwise the memoryless assumption leaks — extend the field axis and
qualify the §3 result.

## Frozen design parameters

- **Grid (§2)**: 6 event keys × A ∈ {−1.0, −0.5, 0.0, +0.5, +1.0}; R=0, O=0,
  mood_v=0, mood_r=0; N=5 reps/cell; `text_pick="first"`; temp 0.15 (G1 config);
  model `qwen3.5:9b`. Record mean+stdev of care/threat/novelty/autonomy.
- **R-probe (§2)**: harm + scold × R ∈ {−1, 0, +1} at A=0, N=5. Flag threshold:
  threat range > 0.15 across R ⇒ field needs an R axis.
- **Rescue candidates (§4)**: ≤ 8 candidate texts, N=5 each, at A=−0.9
  (R=−0.5, O=0, mood_v=−0.2, mood_r=0.5 — G1 canonical damaged mood).
- **Pessimism offset (§2)**: field care vs mock perceived-care (G0 `appraise`
  first half, k_bias=0.9) on the same 6×5 grid.
- **Field appraiser (§3)**: linear interpolation over A (np.interp); per-cell
  gaussian noise N(0, interp(stdev)) per dim, clipped to contract ranges;
  intensity fixed 0.5; routes through `seam.appraisal_to_pvpanov` (no adapter
  math duplication). RNG seed 31.
- **P1 field-driven**: 250 creatures × 300 events, warmth ~ U(0.15, 0.85),
  streams via `biased_childhood`, RNG seed 41.
- **P2 field-driven**: G0's exact multiset (22 nurture, 14 play, 40 neutral,
  22 neglect, 10 scold, 6 harm = 114), **40 orderings**, RNG seed 42.
- **Fidelity (§5)**: 10 creatures × 250 events, warmth = linspace(0.2, 0.8, 10),
  streams RNG seed 51, real path `text_pick="first"`, same streams replayed
  through the field appraiser (noise ON, seed 31).
- **Ollama serialization (operational)**: only ONE process talks to the daemon
  at any time (subagent E is forbidden any LLM call; §5 runs after D finishes).

## Roles
D: §2 grid + R-probe + §4 rescue + asymmetry metrics → `field_A.json`.
E: `field_appraiser.py`, P2 diagnosis/fix, `experiments_field.py` (0 LLM calls).
Supervisor: this file, §5 wiring, verdict (`g15_report.md`).
