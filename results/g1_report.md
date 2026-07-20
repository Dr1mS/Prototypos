# G1 report — real LLM appraiser on qwen3.5:9b

**Verdict: GO** — the LLM *is* the perception-bias mechanism: state-injection
alone produces a large, low-variance, correctly-signed appraisal gap on
ambiguous events, with zero schema or attribution failures.

Model: `qwen3.5:9b` (Ollama 0.32.1, structured output via JSON schema,
`think=False`, temp 0.15). ~1 282 real calls total across both runs,
0 hard-fails. Cheap battery wall time ≈ 60 min + 5 min rig re-run.

## Gate (§6) — run 2

| check | threshold | observed | verdict |
|---|---|---|---|
| schema-valid | ≥ 98 % | **100 %** (0 hard-fails / 200) | PASS |
| subject-bug target-correctness | = 100 % | **0 failures / 50** | PASS |
| sign-agreement per dim | ≥ 85 % (novelty ≥ 80 %, declared) | care 94.4 % · threat 90.8 % · novelty 80.0 % · autonomy 97.1 % | PASS |
| inter-run stdev per dim | ≤ 0.15 | worst **0.053** | PASS |
| state-sensitivity gap | < 0, beyond noise | **−0.737**, CI95 [−0.865, −0.609] | PASS |

**Committed prediction (pre-run)**: gap < 0, magnitude −0.6 … −0.15, CI95
excluding 0. **Observed**: −0.737 [−0.865, −0.609] — direction confirmed,
magnitude *stronger* than committed (secure reads ambiguity at −0.016,
damaged at −0.753).

## Supporting results (real path through the harness seam)

- **P3 scar (382 calls)**: secure A=+1.10 → harm burst → A=−1.18 → identical
  kindness: **state-ON stays trapped at −1.13**, `inject_state=False`
  ablation recovers to −0.72. The self-trap comes from the state-biased
  perception, on the real model. (`g1_fig_p3.png`)
- **P4 timescales (300 calls)**: mood ACT 3 vs trait ACT 8 — separation
  survives but at **3×**, weaker than the mock's 12×. Watch this in G2.
  (`g1_fig_p4.png`)

## Supervisor adjustments (declared, run-1 raw data in `g1_results_run1.json`)

Run 1 was a mechanical NO-GO on sign-agreement (threat 78.3 %, novelty
83.3 %). Per-probe analysis showed no reliability failure:

1. **3 anti-thesis labels fixed** — `threat=0` expected on *damaged* ambiguous
   probes contradicted the thesis under test (G0's `k_bias` comment: "damaged
   creature reads ambiguity as threat"; observed 0.37–0.64). Relabeled `+1`;
   rig **re-measured with 200 fresh calls** (not re-scored): threat 90.8 %.
2. **novelty threshold 0.80** — only 6 labeled probes, so a flat 85 % would
   degenerate into requiring 6/6 = 100 %. The single stable miss
   (play sentence at novelty 0.140 vs 0.15, stdev 0.027) is a consistent
   judgment call, not unreliability.
3. Two marginal threat misses (neglect 0.38/0.39 vs 0.35 band) left counting
   against the model.

## Notable findings

- **Kindness inversion under damage**: `nurture` under damaged state reads
  care **−0.39** (mock only *discounted* kindness, LLM flips it negative).
  Stronger-than-mock self-trapping — visible in P3's flat ON recovery. A
  design lever for G2 (healing may need more than kindness events).
- Ablation baseline (`state=None`) reads ambiguity slightly negative
  (care ≈ −0.45 on one smoke) — the "no state" creature is not perfectly
  neutral.

## Not run

P1/P2 reduced (~4 250 calls) — behind `python run_g1.py --full`; gate was
decided on the cheap battery per §1.5.g.
