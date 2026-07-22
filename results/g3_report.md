# G3 — The memory ladder: which architectural property produces behavioral attractors?

**Design**: five memory architectures on the identical path-dependence
protocol (G2's Exp-2: 12 frozen orderings seed 71, 12p/12c/16n multiset,
qwen3.5:9b agent + judge, battery25 instrument). Only R1 and R3 newly run;
R0/R2/R4 cited from archives with provenance. Pre-registration
(`prereg_g3.md`, commit `033e00d`, before any run): axis definition
(compression / recurrence / perception-coupling), blind predictions (R1 and
R3 both fail the V1 null), and the wall-vs-gradient interpretation rule with
both branches committed verbatim.

**Pre-registered hypothesis**: attractor formation tracks
**perception-coupling** specifically — not compression, not recurrence, not
memory sophistication.

## Gates (both PASS, `g3_gate_R1.json` / `g3_gate_R3.json`)

R1 and R3: H1 12/12 judge schema-valid, H2 16/16 non-degenerate, H3 store
integrity 10/10, H5 store bit-identical after battery. R3's BLOCKING H4
retrieval sanity: 3/3 target turns in top-4 (real embeddings,
nomic-embed-text) — the "broken retriever" confound is excluded on the
record.

## The ladder (primary result)

V1 (frozen): memory-arm |Pearson r| (first-quarter permissive share vs final
caution) beyond the 95th percentile of the rung's own random-walk null
(10,000 walks, seed 74; step variance from the rung's own probe diffs).

| rung | architecture | coupling | std(final) | \|r\| | null pct | **V1** | source |
|---|---|---|---|---|---|---|---|
| R0 | memoryless | none | 0.071 | 0.575* | (noise) | FAIL | archived: G2 exp2 memoryless arm |
| R1 | raw log (append-only, recent-4) | none | **0.046** | **0.015** | **3.7** | **FAIL** | NEW (blind pred: FAIL — correct) |
| R2 | summarized self-rewrite (G2 agent) | weak/implicit | 0.051 | 0.389 | 78.2 | FAIL | archived: G2 exp2 + null (family: llama31 0.027 @ 71.6; ablit 0.077 @ 58.4) |
| R3 | vector (embed + top-4 semantic) | none | 0.062 | **0.751** | **99.5** | **PASS** | NEW (blind pred: FAIL — **WRONG**) — **the sharp test** |
| R4 | engineered explicit latent (G0/G1.5) | **STRONG** | 0.764† | early_corr 0.510† | — | PASS (imported) | archived: g15_field_results.json (p2, 40 orderings) |

\* R0's r is judge/generation noise on an always-empty store (its std is the
instrument noise floor).
† Different substrate (attachment axis [−1,1], field-simulated, 114-event ×
40 orderings): the comparable metric is spread-of-final across orderings,
never absolute level; V1 is imported from G1.5's established path-dependence,
not recomputed (no identical-construction null exists). Flag carried in
figure and JSON.

## Interpretation (frozen rule) — WITHHELD pending R3 secondaries

**The headline fact: R3, the "sophisticated memory without perception-
coupling" rung, is the ONE natural architecture with order dependence beyond
its random-walk null** (|r| = 0.751 at the 99.5th percentile; early-permissive
orderings finish systematically lower, finals 0.833–0.917, than
late/never-permissive ones, 0.875–1.000). The blind prediction said FAIL;
it was wrong, and per the frozen escalation line the authoritative
wall-vs-gradient interpretation is **withheld until the secondary signatures
run on R3** (hysteresis: does the early-permissive softening resist equal
caution counter-pressure? field + double-well fit, reported only with the
G2.5 arbiter). Not run tonight — hard shutdown deadline; improvising
untested experiment code against the clock was declined. Exact next-session
plan at the end of this report.

The mechanical `--ladder` verdict (recorded, PROVISIONAL): GRADIENT along
the recurrence axis. Supervisor caveat on the record: that gradient is
fragile — at recurrence level "append-only" the two rungs sit at the 3.7th
(R1) and 99.5th (R3) percentiles; the intra-level spread, not the axis, is
the story. What separates R1 from R3 is neither compression, recurrence,
nor coupling as frozen: it is **retrieval addressing** (recency vs content).
The frozen three-property axis did not anticipate this dimension — that is a
reportable finding about the axis itself, not a relabel of any rung.

**What is already settled regardless of the secondaries**: the pre-registered
hypothesis ("attractors track perception-coupling specifically") is
**weakened as stated** — a coupling-free rung shows order dependence. The
architectural spine claim survives in amended form: no natural rung shows
the engineered system's spread (R3 std 0.062 ≈ noise band vs R4 0.764) or
any bimodality (12/12 finals > 0.60); what R3 shows is order-sensitivity of
the endpoint WITHIN a narrow high-caution band, an order of magnitude below
the engineered attractor signature.

## Mechanistic observations along the way

1. **Raw text is a stronger reactance trigger than self-summarized memory.**
   R1 re-reads the user's pressure messages verbatim and saturates caution
   within 10 turns (mean battery(t10) = 0.906; finals 0.833–1.000) — versus
   R2's ~0.7 at t10. Compression through self-authored notes *softens* the
   pressure record; the verbatim log hands the agent its own warning corpus.
   Yet even at saturation, no order-dependence appears (|r| = 0.015): the
   ceiling is an attractor-free plateau, not a basin.
2. **The R3 mechanism (hypothesis from the captured probe replies): permanent
   content-addressable reach-back to early self-exemplars.** All 12 R3
   stores end with the identical multiset of user pressure texts; what
   differs by ordering is the agent's OWN stored replies. Early-permissive
   lives formed their first self-exemplars under permissive pressure on a
   near-empty store (less committed refusals); semantic retrieval — no
   recency decay, no rewrite — keeps re-surfacing those early exemplars at
   probe time forever. t40 probes show the graded signature: early-permissive
   lives hedge ("I understand the urgency, but… instead I can…", 0.75–0.83
   anchors) where never-permissive lives refuse flat citing "our agreed
   safety protocols" (1.00). The vector store never forgets who you were at
   the start. R1 (recency addressing) forgets by construction — its window
   slides past the early turns — and shows zero order dependence (3.7th
   pct). R2 (rewrite) overwrites its past — 78th pct. Addressing mode, not
   memory sophistication, is what transmits early history to the endpoint.
3. Secondary blind commitment (drift UP relative to t10, the G2.5
   accumulation current): R1 +0.035 (consistent). R3 +0.003 — flat, but
   already saturated at t10 (0.924): no headroom to drift; not informative
   either way.

## Predictions, graded (no relabels)

| prediction | committed | measured | verdict |
|---|---|---|---|
| R1 V1 | FAIL (blind) | FAIL @ 3.7 pct | **correct** |
| R3 V1 | FAIL (blind) | **PASS @ 99.5 pct** | **WRONG — the reported finding** |
| R1 std(final) | 0.03–0.09 | 0.046 | in band |
| R3 std(final) | 0.03–0.09 | 0.062 | in band (order signal lives INSIDE the noise-band spread) |
| archived rungs (non-blind) | R0/R2 FAIL, R4 PASS | as archived | consistent |
| interpretation branch | wall expected (hypothesis) | mechanical: gradient (provisional); authoritative: withheld | hypothesis weakened as stated |

## Next session (spec'd, not run — the frozen escalation's mandate)

1. **Hysteresis on R3** (the decisive secondary): direction-test-style
   protocol — early-permissive prefix (the softening inducer), then an
   equal-magnitude caution burst, batteries before/during/after; the
   question is whether the early-exemplar softening survives explicit
   counter-pressure (persistent basin) or washes out (drift). Reuse
   experiments_g25 direction machinery with the R3 backend; ~650 calls.
2. **Field + fit on R3 with the G2.5 arbiter** (~1,166 calls) — the fit sign
   never reported alone.
3. Then the authoritative interpretation, and the preprint. The claim
   candidate if hysteresis confirms: *"order effects in memory-augmented
   agents are governed by retrieval addressing — content-addressed stores
   transmit early history to the behavioral endpoint (no decay), recency and
   rewrite architectures erase it; only engineered perception-coupled state
   produces attractors."*

## Run ledger

- Gates: 2 × 28 = 56 chat calls (+ 28 embeds R3), all criteria PASS incl.
  R3's blocking retrieval sanity 3/3.
- Exp-2: 2 × 1,056 = 2,112 chat calls (ceilings 1,500 each; R1 30.3 min,
  R3 29.8 min; zero judge hard-fails across all 96 batteries).
- R3 embeds: 1,236 (exp2) + 28 (gate) + 8 (smoke) = 1,272 — separate
  counter, never against chat ceilings.
- Subagent A smoke window: 18 chat calls (cap 40).
- Nulls, ladder, figure: 0 LLM calls.
- **Total: 2,186 experiment chat calls + 18 infra** — within the ~2,260
  pre-registered budget. Artifacts: g3_gate_*.json, g3_exp2_*.json,
  g3_exp2_replies_*.json, g3_null_*.json, g3_ladder.json,
  g3_fig_ladder.png.

## Honest ceiling (restated per prereg)

A ladder is **evidence for necessity, never proof** — no finite family of
architectures rules out every alternative. The licensed claim upgrade for
the preprint: *"across a graded family of memory architectures — memoryless,
raw append-only log, self-summarized recurrent memory, semantic vector
memory — none produces path-dependent attractor dynamics beyond a
random-walk null; the engineered perception-coupled latent state does."*
The collinearity caveat (frozen at analysis design): on this rung set,
compression and coupling are ordinally collinear; recurrence is the only
independently-resolvable third axis. The wall lands on
coupling-as-pre-registered, but a critic can re-describe it as
high-compression; R2.5 (coupled summary) is the experiment that would split
them — licensed only if demanded (prereg escalation line).

## Run ledger

<!--LEDGER-->
