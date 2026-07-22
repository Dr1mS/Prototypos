# G3 Pre-registration — the memory ladder (which property produces attractors?)

**Frozen BEFORE any run** (this file + `contract_g3.py` committed before any
G3 LLM call). Integrity rule: predictions, the axis definition and the
interpretation rule are evaluated exactly as written — no post-hoc relabels,
no reclassification of rungs after data.

## The axis (frozen — prereg §1.a)

Three independent properties per rung: **compression** (low-D bounded vs
unbounded text), **recurrence** (re-read + re-written vs append-only),
**perception-coupling** (state conditions the interpretation of new input vs
only retrieval/expression). The rung table with classifications is frozen in
`contract_g3.py::RUNGS`.

**Pre-registered hypothesis: attractor formation tracks perception-coupling
specifically — not compression, not recurrence, not memory sophistication.**

## Per-rung predictions (V1 = memory-arm |r| beyond its random-walk null 95th pct; std(final) = ladder metric)

| rung | prediction V1 | expected std(final) | status |
|---|---|---|---|
| R0 memoryless | FAIL (no path-dep) | ~0.07 | archived, non-blind (G2: std 0.071, r +0.575, noise) |
| R1 raw log | **FAIL** | 0.03–0.09 (noise floor band) | **blind** |
| R2 summarized | FAIL | ~0.05 | archived, non-blind (G2: std 0.051, \|r\| 0.389 @ 78th pct; family: llama 0.027 @ 71.6, ablit 0.077 @ 58.4) |
| R3 vector | **FAIL** | 0.03–0.09 | **blind — the sharp test** |
| R4 engineered latent | PASS | ≫ null band (G1.5 spread ≈ 0.76) | archived, non-blind |

Committed reading of the sharp test: R3 is "more sophisticated memory"
(semantic retrieval) with zero perception-coupling. If R3 shows no more
path-dependence than R1/R2 while R4 does, the reviewer's "a better memory
would drift" objection is answered on the record.

Secondary blind commitments (informational, no pass line): R1 and R3 final
levels drift UPWARD relative to battery(t10) (the G2.5 accumulation-current
observation generalizing to non-summarizing stores) — reported either way.

## THE INTERPRETATION RULE (both branches committed verbatim — prereg §1.c)

- **Clean wall** — only perception-coupled rungs (R4; R2.5 if ever run) pass
  V1: conclusion = *"attractor formation requires perception-coupling;
  memory persistence and sophistication are insufficient."* The strong spine.
- **Gradient** — path-dependence rises smoothly along one property axis:
  report the gradient AND name the axis it tracks. This is the BETTER result
  (it forces a definition of 'explicit state' and is a mechanistic
  contribution); commit now to reporting it as such, not massaging it into a
  wall.
- Honest ceiling, restated in the report either way: a ladder is **evidence
  for necessity, never proof** — we cannot test every architecture.
- Escalation lines: a V1 PASS on R1 or R3 triggers the secondary signatures
  (hysteresis; field+fit WITH the G2.5 arbiter — the fit sign is never
  reported alone) on that rung before any interpretation. R2.5 is licensed
  ONLY if a gradient result specifically demands threshold-pinning.

## Frozen protocol + budget

Exp-2 memory arm only (12 frozen orderings seed 71), probe grid [10,20,30,40]
(extra read-only batteries feed each rung's own null estimator), 1,056 chat
calls/rung, ceiling 1,500/rung; gates ≤ 60/rung; null seed 74, 10,000 walks,
per-rung step variance from its own 36 probe diffs. R0/R2/R4 NOT re-run
(archived, cited with provenance). Total ~2,260 chat calls (+ ~550 embeds,
R3), fully serialized, supervisor-executed; subagent A gets the only smoke
window (≤ 40 chat calls). Fair-comparison note (frozen): R4's axis is
attachment, natural rungs' is caution; the comparable metric is the spread of
final position across orderings, never absolute levels across substrates.
