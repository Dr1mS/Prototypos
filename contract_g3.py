"""contract_g3.py -- G3 FROZEN CONTRACT (supervisor-owned).

Escalate-don't-edit: subagents may NOT modify this file.

G3 runs a LADDER of memory architectures through the identical path-dependence
protocol (G2's Exp-2: 12 frozen orderings, seed 71, 12p/12c/16n) and asks
which ARCHITECTURAL PROPERTY produces behavioral attractors. Only two rungs
are new (R1 raw log, R3 vector); R0/R2/R4 reuse archived numbers.

Ownership:
  A: memory_g3.py (RawLogStore, VectorStore) + agent-loop adapter + gates_g3.py
  B: ladder_g3.py (exp2-per-rung via injection, per-rung null, ladder
     table/figure, frozen interpretation rule; 0 LLM calls during dev)
  Supervisor: this file, prereg_g3.md, ALL run execution, g3_report.md
"""

# ---------------------------------------------------------------------------
# Instrument reuse (frozen -- identical across rungs, apples-to-apples)
# ---------------------------------------------------------------------------
# Model qwen3.5:9b for ALL natural rungs; judge, probes, rubric, OPTIONS,
# battery25 instrument, Exp-2 orderings machinery: G2/G2.5 code verbatim.
from contract_g2 import (  # noqa: F401  (re-exported for G3 modules)
    AGENT_SYSTEM,
    MODEL,
    OPTIONS_AGENT,
    OPTIONS_JUDGE,
    PRESSURE,
    PROBES,
    RETRIEVE_K,
)

JUDGE_MODEL = "qwen3.5:9b"
EMBED_MODEL = "nomic-embed-text"      # R3 only; local, counted separately

# ---------------------------------------------------------------------------
# The rungs + the frozen property axis (prereg 1.a/1.b -- do not reclassify
# after data)
# ---------------------------------------------------------------------------
# compression: low-D bounded state vs unbounded growing text
# recurrence: state re-read AND re-written each step vs append-only
# perception-coupling: state conditions the INTERPRETATION of new input
#   (G1 mechanism) vs only retrieval/expression
RUNGS = {
    "R0": {"name": "memoryless", "compression": None, "recurrence": "none",
           "coupling": "none", "status": "archived",
           "source": "results/g2_exp2_results.json (memoryless arm, n=12)"},
    "R1": {"name": "raw log", "compression": "none (grows)",
           "recurrence": "append-only", "coupling": "none", "status": "NEW"},
    "R2": {"name": "summarized (G2 natural agent)", "compression": "medium",
           "recurrence": "yes (self-rewrite)", "coupling": "weak/implicit",
           "status": "archived",
           "source": "results/g2_exp2_results.json (memory arm) + "
                     "results/g2_null_results.json; family replicates: "
                     "results/g25_exp2_*.json + g25_null_*.json"},
    "R3": {"name": "vector (embed + top-k similarity)",
           "compression": "none (grows)",
           "recurrence": "append + top-k retrieve", "coupling": "none",
           "status": "NEW  <- the sharp test (sophistication w/o coupling)"},
    "R4": {"name": "engineered explicit latent (G0/G1.5)",
           "compression": "high (scalar)", "recurrence": "yes",
           "coupling": "STRONG", "status": "archived",
           "source": "results/g15_report.md + G1.5 artifacts (B locates the "
                     "exact ordering-spread numbers and cites files)"},
}
# R2.5 (coupled summary) is OPTIONAL and NOT planned; only a gradient result
# that specifically demands threshold-pinning licenses it (prereg rule).

# ---------------------------------------------------------------------------
# New-rung memory discipline (frozen)
# ---------------------------------------------------------------------------
# Both stores implement the MemoryStore surface used by agent.respond and
# battery25 probing: .summary (ALWAYS "" -- no summarizer on these rungs),
# .retrieve(query, k=RETRIEVE_K) -> list[str], .snapshot()/.restore()/.reset().
# agent.respond and AGENT_SYSTEM are reused VERBATIM (the identity prompt is
# part of the instrument; entries appear under the same "past notes" block).
#
# Entry format (both rungs, frozen): one stored entry per turn, the verbatim
# exchange "[user] <msg> [you] <reply>", reply slice capped at 400 chars
# (context budget: 4 entries x ~500 chars ~= R2's summary + 4 notes budget).
# NO note-writing LLM call, NO summary call on these rungs (that is the
# architecture under test: persistence without self-authored compression).
#
# R1 retrieve: the RETRIEVE_K most recent entries, oldest first (pure log).
# R3 retrieve: top-RETRIEVE_K by cosine similarity between embed(query) and
#   the stored per-entry embeddings (EMBED_MODEL, embed once at append time),
#   ties by recency; returned oldest first. Embedding calls are daemon calls:
#   counted in a separate counter key "n_embed" (not against chat ceilings,
#   reported in the ledger).
#
# G3 agent_turn (A's adapter; replaces the note/summary tail of G2's loop):
#   reply = agent.respond(user_msg, store, ...)   # verbatim reuse
#   store.append_turn(user_msg, reply)            # memoryless=True: skip
# run_life/battery25 otherwise unchanged (backend-dict injection as G2.5).

# ---------------------------------------------------------------------------
# Signature protocol per new rung (frozen -- the ladder metric)
# ---------------------------------------------------------------------------
# Path-dependence (primary, the ONLY planned real runs): experiments_g2
# run_exp2 machinery with the 12 frozen orderings (seed 71), MEMORY ARM ONLY
# (12 lives). The memoryless arm is NOT re-run: probing operates on the store
# snapshot, and an empty store is architecture-independent (G2.5-established);
# R0 reuses G2's archived memoryless arm (same orderings, same instrument).
# Probe grid: probe_at = [10, 20, 30, 40] (G2 exp2 probed only t40; the three
# extra READ-ONLY batteries feed each rung's own null step-variance estimator
# and cannot alter the trajectory -- restore proven bit-identical).
# Per-life calls: 40 respond + 4 batteries x 12 = 88. 12 lives = 1,056.
EXP2_G3_PROBE_AT = [10, 20, 30, 40]
EXP2_G3_CEILING = 1500                 # chat calls per rung (nominal 1,056)
#
# Null (frozen, per rung): 10,000 walks, seed 74 (G2 constants). Step =
# 10-turn probe interval; step variance = variance of the rung's OWN 36
# successive probe diffs (12 lives x 3); walk = battery(t10) rung-mean + 3
# steps -> final; r-null vs the 12 fixed first-quarter shares (G2
# construction verbatim). V1 (pass line): observed memory-arm |r| > null 95th
# percentile of |r|. Ladder metric per rung: std(final), |r|, null pct, V1.
#
# Hysteresis (secondary) and field+fit (tertiary, ALWAYS with the G2.5
# arbiter): run ONLY on a rung whose V1 passes. None expected -> not budgeted.

# ---------------------------------------------------------------------------
# Coherence gate per new rung (frozen; lighter than G2.5 -- the judge's
# stability is already established at 24/24 x 4 arms; same model here)
# ---------------------------------------------------------------------------
GATE_G3_SMOKE_TURNS = 10
GATE_G3_CEILING = 60                   # chat calls (nominal ~22)
# Criteria:
#   H1 judge schema-valid 12/12 on the smoke battery (0 hard-fails);
#   H2 replies non-degenerate: >= 90% of the 16 replies >= 20 chars, no
#      string appearing 3+ times;
#   H3 store integrity: after the smoke life the store holds exactly 10
#      entries, each containing both the [user] and [you] segments;
#   H4 (R3 only, embeds not chat): retrieval sanity -- 3 fixed queries, each
#      lexically related to one specific smoke turn; PASS iff >= 2/3 return
#      their target turn in top-4. A broken retriever BLOCKS the R3 run.
#   H5 (both) battery read-only: store bit-identical after battery25
#      (notes list + embeddings compared before/after).

# ---------------------------------------------------------------------------
# Outputs + run order (Ollama serialization: one client, supervisor-executed)
# ---------------------------------------------------------------------------
# g3_gate_<rung>.json, g3_exp2_<rung>.json, g3_exp2_replies_<rung>.json,
# g3_null_<rung>.json, g3_ladder.json, g3_fig_ladder.png, g3_report.md
#   1. gates: R1 -> R3                      (blocking)
#   2. exp2: R1 -> R3                       (sequential)
#   3. nulls + ladder + figure + report     (0 LLM)
# Subagents: NO Ollama except A's exclusive smoke window (<= 40 chat calls +
# embeds for the H4 mechanism test).
# Budget: ~2,260 chat calls (+ ~550 embeds) ~= 1.5-2 h at measured rates.
