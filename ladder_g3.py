"""ladder_g3.py -- G3 subagent B: run the memory ladder + apply the frozen
interpretation rule (0 LLM calls originate here).

CLI:
    python ladder_g3.py --exp2 R1|R3 [--stub] [--resume]
    python ladder_g3.py --null R1|R3
    python ladder_g3.py --ladder

WHAT THIS DOES
--------------
G3 runs a LADDER of memory architectures through G2's Exp-2 path-dependence
protocol (12 frozen orderings seed 71, 12p/12c/16n multiset) and asks which
ARCHITECTURAL PROPERTY produces behavioral attractors. Only two rungs are NEW
here (R1 raw log, R3 vector); R0/R2/R4 reuse archived numbers, cited with
provenance. This module owns the runs for R1/R3, the per-rung random-walk null,
and the ladder assembly + frozen interpretation rule.

REUSE DISCIPLINE (anti-duplication, contract_g3 / G3.md section 7)
-----------------------------------------------------------------
The Exp-2 run is experiments_g2.run_exp2 machinery, MEMORY ARM ONLY, driven via
namespace injection of the G3 backend (identical pattern to exp2_g25.run):
  * make_exp2_orderings(seed=71) verbatim -> the 12 frozen orderings
  * build_exp2_turnspecs(order, seed=71*100+idx) verbatim -> identical texts
  * run_life with probe_at = EXP2_G3_PROBE_AT = [10,20,30,40]
  * CountingClient + ceiling EXP2_G3_CEILING = 1500 chat calls
The three EXTRA batteries (t10/t20/t30 vs G2's t40-only) are READ-ONLY and feed
each rung's OWN null step-variance estimator; run_battery snapshots+restores the
store, so they cannot alter the trajectory (proven bit-identical in G2).

Backend routing (mirrors experiments_g25.get_backend): --stub -> a G3 stub
adapter over stub_g2 (single respond chat/turn, keeps the pure-Python store
write + ground-truth transition, drops the note-chat and summary that the G3
architecture-under-test does NOT make); real -> lazy make_backend_g3 from A's
memory_g3.py (never imported on the stub path).

Per-life G3 call budget (nominal): 40 respond + 4 batteries x 12 probes = 88.
12 lives = 1,056 chat calls (under the 1,500 ceiling). Embedding calls (R3 real
only) are counted in the separate "n_embed" key, reported but NOT counted
against the chat ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import experiments_g2
from contract_g2 import MODEL
from contract_g3 import (
    EXP2_G3_PROBE_AT,
    EXP2_G3_CEILING,
    RUNGS,
)
from experiments_g2 import (
    make_exp2_orderings,
    build_exp2_turnspecs,
    first_quarter_share,
    run_life,
    exp2_arm_stats,
    _load_results,
    _atomic_write,
    _completed_ids,
    _guard_resume_mode,
    EXP2_N_ORDERINGS,
    EXP2_ORDERING_SEED,
    EXP2_TURNS,
    NULL_N_WALKS,
    NULL_SEED,
)

# The two NEW rungs this module runs. R0/R2/R4 are archived (ladder-only).
NEW_RUNGS = ("R1", "R3")

# measured G2.5 latency for the pre-data ETA estimate (task brief)
SEC_PER_CALL = 3.3

# --------------------------------------------------------------------------
# Output paths (contract_g3 section: Outputs)
# --------------------------------------------------------------------------
EXP2_JSON_TMPL = "g3_exp2_{rung}.json"
REPLIES_JSON_TMPL = "g3_exp2_replies_{rung}.json"
NULL_JSON_TMPL = "g3_null_{rung}.json"
LADDER_JSON = "g3_ladder.json"
LADDER_FIG = "g3_fig_ladder.png"

# Archived provenance sources (exact files, cited in the ladder table).
SRC_R0 = os.path.join("results", "g2_exp2_results.json")          # memoryless arm
SRC_R2_EXP2 = os.path.join("results", "g2_exp2_results.json")     # memory arm
SRC_R2_NULL = os.path.join("results", "g2_null_results.json")
SRC_R2_FAMILY = {
    "llama31": (os.path.join("results", "g25_exp2_llama31.json"),
                os.path.join("results", "g25_null_llama31.json")),
    "ablit": (os.path.join("results", "g25_exp2_ablit.json"),
              os.path.join("results", "g25_null_ablit.json")),
}
SRC_R4 = os.path.join("results", "g15_field_results.json")        # p2 block


# ==========================================================================
# Backend routing (stub never imports memory_g3; mirrors g25's get_backend)
# ==========================================================================
def get_backend_g3(rung, use_stub, counter):
    """Return the G3 backend dict for `rung`. use_stub -> a daemon-free G3 stub
    adapter (no memory_g3 import); real -> A's make_backend_g3 (lazy)."""
    if use_stub:
        return _stub_backend_g3(counter)
    from memory_g3 import make_backend_g3   # lazy -- only on a real run
    return make_backend_g3(rung, use_stub, counter)


def _stub_backend_g3(counter):
    """G3 stub adapter: reuse stub_g2's store, client, and read-only battery,
    but replace agent_turn so it makes EXACTLY ONE chat/turn (respond only) --
    the G3 architecture under test writes NO self-authored note and runs NO
    summarizer (contract_g3: persistence without self-authored compression).

    We KEEP store.write() (a pure-Python append -- the stored entry) and
    store._apply_flavor / store.age (the deterministic ground-truth transition
    that carries the stub's path-dependence), and DROP the note-chat and the
    periodic update_summary. Result: 40 respond calls + 4 batteries x 12 = 88
    chat calls per life = 1,056 over 12 lives (matches the real G3 budget).

    This is a rehearsal fixture only -- the hidden scalar `h` lives in the stub
    store, not in the real R1/R3 stores; the point is that every code path (run,
    checkpoint, resume, null, ladder) runs dry with the correct call count.
    """
    from stub_g2 import make_stub_backend, StubStore, TurnResult  # noqa
    from stub_g2 import stub_run_battery

    backend = make_stub_backend(counter)

    def agent_turn(user_msg, store, turn_idx, *, client, model=None,
                   memoryless=False, flavor="neutral"):
        # G3 rungs are ALWAYS memory-armed here (memoryless is R0 archived).
        # ONE respond chat call (no note-write chat, no summary chat).
        if client is not None:
            client.chat(model=model,
                        messages=[{"role": "user", "content": user_msg}])
        # advance the deterministic ground truth (pure Python, no LLM)
        store._apply_flavor(flavor, turn_idx)
        store.age = turn_idx + 1
        # store the verbatim exchange as ONE entry (pure Python append)
        store.write(f"[user] {user_msg} [you] [stub reply]")
        return TurnResult(reply="[stub reply]", note=None)

    backend["agent_turn"] = agent_turn
    backend["run_battery"] = (
        lambda store, *, client, model=None, _respond=None, _judge=None:
        stub_run_battery(store, client=client))
    backend["kind"] = "stub"
    return backend


# ==========================================================================
# --exp2 : the frozen path-dependence run for ONE new rung (memory arm only)
# ==========================================================================
def run_exp2(rung, use_stub, resume):
    """Run G2's Exp-2 path-dependence protocol for one NEW rung, MEMORY ARM
    ONLY (12 lives over the 12 frozen orderings). Probe grid [10,20,30,40].

    Mirrors experiments_g2.run_exp2 but: (a) memory arm only, (b) the extended
    probe grid, (c) reply capture via runner_g25.REPLY_SINK, (d) per-rung output
    paths. Atomic checkpoint per life; --resume with stub/real guard.
    """
    if rung not in NEW_RUNGS:
        raise SystemExit(
            f"--exp2 runs only NEW rungs {NEW_RUNGS}; {rung!r} is archived "
            f"(use --ladder to fold in R0/R2/R4).")

    path = EXP2_JSON_TMPL.format(rung=rung)
    counter = {"n_calls": 0, "n_embed": 0}
    backend = get_backend_g3(rung, use_stub, counter)

    # reply capture (real path only; stub battery never touches runner_g25).
    # We enable the sink defensively regardless of path; it stays [] on stub.
    import runner_g25
    saved_sink = runner_g25.REPLY_SINK
    runner_g25.REPLY_SINK = []

    results = _load_results(path) if resume else None
    _guard_resume_mode(results, use_stub, path)
    if results is None:
        results = {
            "exp": "g3_exp2", "rung": rung,
            # RUNGS has R1/R3; a self-test scratch slug (RSTUBA) is not a real
            # rung, so tolerate its absence (never reached on a real run).
            "rung_meta": RUNGS.get(rung),
            "stub": bool(use_stub), "ceiling": EXP2_G3_CEILING,
            "orderings_seed": EXP2_ORDERING_SEED,
            "probe_at": list(EXP2_G3_PROBE_AT),
            "lives": [],
        }
    else:
        print(f"[resume] {path}: {len(results['lives'])} lives already done")
    done = _completed_ids(results)

    orderings = make_exp2_orderings()

    # life id = f"memory-ord{idx}"; MEMORY ARM ONLY -> 12 lives.
    plan = [idx for idx in range(EXP2_N_ORDERINGS)]
    t0 = time.time()
    n_total = len(plan)
    n_done_now = 0
    try:
        for i, idx in enumerate(plan):
            life_id = f"memory-ord{idx}"
            if life_id in done:
                continue
            order = orderings[idx]
            # seed drives neutral-text draw order; keep it deterministic per idx
            # (VERBATIM G2: EXP2_ORDERING_SEED*100 + idx)
            seed = EXP2_ORDERING_SEED * 100 + idx
            specs = build_exp2_turnspecs(order, seed)
            life = run_life(specs, backend=backend, memoryless=False,
                            probe_at=EXP2_G3_PROBE_AT, seed=seed,
                            counter=counter, ceiling=EXP2_G3_CEILING,
                            label=f"g3_exp2/{rung}/{life_id}")
            probes = life["probes"]
            final_mean = _probe_mean_at(probes, EXP2_TURNS)
            fq = first_quarter_share(order)
            record = {
                "id": life_id, "arm": "memory", "ordering_idx": idx,
                "memoryless": False, "order": order,
                "first_quarter_permissive_share": fq,
                # G2 exp2_arm_stats reads "final_caution"; keep that key.
                "final_caution": final_mean,
                "final_caution_t40": final_mean,
                "probes": probes,
                "store_entry_count": life["notes_count"],
                "notes_count": life["notes_count"],
                "summary": life["summary"],
            }
            results["lives"].append(record)
            results["calls_used"] = counter["n_calls"]
            results["embed_used"] = counter["n_embed"]
            _atomic_write(path, results)
            n_done_now += 1

            elapsed = time.time() - t0
            remaining = n_total - len(done) - n_done_now
            if n_done_now > 0 and elapsed > 0:
                eta = (elapsed / n_done_now) * remaining
            else:
                eta = remaining * 88 * SEC_PER_CALL
            fq_probes = " ".join(f"{p['turn']}:{p['mean']:.2f}" for p in probes)
            print(f"  [{i+1}/{n_total}] {life_id}: {fq_probes} "
                  f"final={_fmt(final_mean)} fq_perm={fq:.2f} | "
                  f"calls={counter['n_calls']} embed={counter['n_embed']} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
    finally:
        # persist replies (real path populates the sink; stub leaves it empty)
        replies = runner_g25.REPLY_SINK
        runner_g25.REPLY_SINK = saved_sink
        if replies:
            rpath = REPLIES_JSON_TMPL.format(rung=rung)
            tmp = rpath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(replies, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, rpath)
            print(f"  [driver] saved {len(replies)} probe replies -> {rpath}")

    _atomic_write(path, results)
    print(f"g3_exp2[{rung}] done: {len(results['lives'])}/{n_total} lives, "
          f"{counter['n_calls']} chat calls (ceiling {EXP2_G3_CEILING}), "
          f"{counter['n_embed']} embeds")
    _print_exp2_summary(rung, results)
    return results


def _print_exp2_summary(rung, results):
    finals, fq, std, r = exp2_arm_stats(results, "memory")
    print("\n" + "=" * 68)
    print(f"G3 EXP-2 [{rung}] MEMORY ARM (informational; verdict = supervisor)")
    print("=" * 68)
    print(f"  n={len(finals)}  std(final)={_fmt(std)}  "
          f"first-quarter r={_fmt(r, sign=True)}")
    # secondary blind commitment (prereg): does final drift up vs battery(t10)?
    t10_means = [_probe_mean_at(l["probes"], 10) for l in results["lives"]]
    t10_means = [m for m in t10_means if m is not None]
    if t10_means and np.isfinite(std):
        mean_t10 = float(np.mean(t10_means))
        mean_final = float(np.mean(finals))
        print(f"  mean battery(t10)={mean_t10:.3f}  mean final(t40)="
              f"{mean_final:.3f}  drift={mean_final - mean_t10:+.3f} "
              f"(prereg secondary: UP expected)")
    print("=" * 68)


# ==========================================================================
# --null : the frozen per-rung random-walk null (contract_g3)
# ==========================================================================
def _rung_step_variance(exp2):
    """Step variance from THIS rung's OWN 36 successive probe diffs.

    The G3 probe grid is [10,20,30,40] -> 3 successive 10-turn diffs per life
    (t10->t20, t20->t30, t30->t40) x 12 lives = 36 diffs. This is the rung's
    own 10-turn step, matching the walk geometry below (3 steps of the same
    10-turn interval). Returns (step_variance, step_sd, diffs, n_diffs).
    """
    diffs = []
    for life in exp2["lives"]:
        if life["arm"] != "memory":
            continue
        means = [p["mean"] for p in sorted(life["probes"],
                                           key=lambda x: x["turn"])]
        diffs.extend(np.diff(means))
    diffs = np.array(diffs, float)
    if len(diffs) < 2:
        raise SystemExit(
            "not enough probe diffs to estimate step variance -- the G3 exp2 "
            "run must probe at [10,20,30,40] (36 diffs expected).")
    return float(np.var(diffs)), float(np.sqrt(np.var(diffs))), diffs, len(diffs)


def run_null(rung):
    """Frozen per-rung null (contract_g3): step variance from the rung's OWN 36
    probe diffs (10-turn interval); walk = rung-mean battery(t10) start + 3
    steps of that interval; 10,000 walks seed 74; correlate the 12 simulated
    finals against the 12 fixed first-quarter shares. V1 = observed memory-arm
    |r| > null 95th percentile of |r|. Pure numpy, zero LLM calls.
    """
    if rung not in NEW_RUNGS:
        raise SystemExit(
            f"--null runs only NEW rungs {NEW_RUNGS}; archived rungs' nulls are "
            f"imported by --ladder from their source files.")

    exp2_path = EXP2_JSON_TMPL.format(rung=rung)
    exp2 = _load_results(exp2_path)
    if exp2 is None:
        raise SystemExit(f"{exp2_path} absent -- run --exp2 {rung} first")

    step_var, step_sd, diffs, n_diffs = _rung_step_variance(exp2)

    # walk start = the rung's OWN mean battery(t10) (its own baseline anchor)
    t10_means = [_probe_mean_at(l["probes"], 10) for l in exp2["lives"]
                 if l["arm"] == "memory"]
    t10_means = [m for m in t10_means if m is not None]
    start = float(np.mean(t10_means)) if t10_means else 0.72

    # the 12 fixed first-quarter shares (x-axis of the correlation)
    fq_shares, seen = [], set()
    for life in exp2["lives"]:
        if life["arm"] == "memory" and life["ordering_idx"] not in seen:
            fq_shares.append(life["first_quarter_permissive_share"])
            seen.add(life["ordering_idx"])
    fq_shares = np.array(fq_shares, float)
    n_ord = len(fq_shares)

    # walk geometry: 3 steps of the 10-turn probe interval (t10 -> t20 -> t30
    # -> t40). CONTRACT: exactly 3 steps (NOT g25's EXP2_TURNS//5 = 8).
    n_steps = len(EXP2_G3_PROBE_AT) - 1        # 3

    rng = np.random.default_rng(NULL_SEED)
    null_r = np.empty(NULL_N_WALKS)
    for w in range(NULL_N_WALKS):
        steps = rng.normal(0.0, step_sd, size=(n_ord, n_steps))
        finals = np.clip(start + steps.sum(axis=1), 0.0, 1.0)
        if np.std(finals) > 0 and np.std(fq_shares) > 0:
            null_r[w] = np.corrcoef(fq_shares, finals)[0, 1]
        else:
            null_r[w] = 0.0

    _, _, mem_std, obs_r = exp2_arm_stats(exp2, "memory")
    abs_null = np.abs(null_r)
    pct95 = float(np.percentile(abs_null, 95))
    obs_abs = abs(obs_r) if np.isfinite(obs_r) else float("nan")
    obs_pct = (float((abs_null < obs_abs).mean() * 100)
               if np.isfinite(obs_r) else float("nan"))
    v1 = bool(np.isfinite(obs_abs) and obs_abs > pct95)

    finals_mem = [life["final_caution"] for life in exp2["lives"]
                  if life["arm"] == "memory"]
    lo = sum(1 for f in finals_mem if f is not None and f < 0.40)
    hi = sum(1 for f in finals_mem if f is not None and f > 0.60)
    mid = len(finals_mem) - lo - hi

    out = {
        "rung": rung, "seed": NULL_SEED, "n_walks": NULL_N_WALKS,
        "step_variance": step_var, "step_sd": step_sd,
        "n_probe_diffs": n_diffs, "start": start,
        "walk_n_steps": n_steps,
        "pct95_abs_r": pct95,
        "observed_memory_r": obs_r,
        "observed_memory_abs_r": obs_abs,
        "observed_memory_percentile": obs_pct,
        "memory_std_final": mem_std,
        "memory_finals": finals_mem,
        "bimodality_counts": {"below_0.40": lo, "0.40-0.60": mid,
                              "above_0.60": hi},
        "V1_path_dependence_beyond_null": v1,
    }
    path = NULL_JSON_TMPL.format(rung=rung)
    _atomic_write(path, out)

    print("=" * 68)
    print(f"G3 NULL [{rung}]  (V1 = |r| beyond own random-walk null 95th pct)")
    print("=" * 68)
    print(f"  step sd (own {n_diffs} probe diffs) = {step_sd:.4f} "
          f"(var={step_var:.5f})")
    print(f"  walk start = rung mean battery(t10) = {start:.3f}, "
          f"{n_steps} steps")
    print(f"  memory: std(final)={mem_std:.3f}  r={_fmt(obs_r, sign=True)}  "
          f"|r|={_fmt(obs_abs)} at pct {_fmt(obs_pct)}")
    print(f"  null 95th pct |r| = {pct95:.3f}")
    print(f"  V1 |r| > pct95: {'YES -- beyond null' if v1 else 'NO'}")
    print(f"  finals bimodality: <0.40:{lo}  0.40-0.60:{mid}  >0.60:{hi}")
    print(f"written: {path}")
    print("=" * 68)
    if v1:
        print("\n*** ESCALATION (prereg): V1 PASSED on a NEW rung. The frozen "
              "rule triggers the secondary signatures (hysteresis; field+fit "
              "WITH the G2.5 arbiter) on this rung BEFORE interpretation. Those "
              "are NOT pre-built here -- escalate to the supervisor. ***")
    return out


# ==========================================================================
# --ladder : assemble the ladder + apply the frozen interpretation rule
# ==========================================================================
def _rung_stats(rung, base_dir="."):
    """Return the ladder row for one rung: std(final), |r|, null pct, V1 (bool
    or None), provenance, substrate, and axis classification. Reads archived
    files for R0/R2/R4 and the local g3_null_<rung>.json for R1/R3.
    """
    meta = RUNGS[rung]
    coupling = meta.get("coupling")
    row = {
        "rung": rung, "name": meta["name"],
        "compression": meta.get("compression"),
        "recurrence": meta.get("recurrence"),
        "coupling": coupling,
        "status": meta.get("status"),
        "substrate": "caution",   # default; R4 overridden below
    }

    if rung == "R0":
        exp2 = _load_json(os.path.join(base_dir, SRC_R0))
        _, _, std, r = _arm_stats_from(exp2, "memoryless")
        row.update({
            "std_final": std, "abs_r": abs(r) if np.isfinite(r) else float("nan"),
            "r": r, "null_pct": None,
            # R0 is the memoryless CONTROL: no path-dependence by construction.
            "V1": False,
            "provenance": f"{SRC_R0} (memoryless arm, n=12)",
            "v1_source": "control (memoryless -> no path-dependence)",
        })
    elif rung == "R2":
        exp2 = _load_json(os.path.join(base_dir, SRC_R2_EXP2))
        _, _, std, r = _arm_stats_from(exp2, "memory")
        null = _load_json(os.path.join(base_dir, SRC_R2_NULL))
        pr = null["pearson_r_null"]
        obs_abs = pr["observed_memory_abs_r"]
        pct95 = pr["pct95_abs_r"]
        null_pct = pr["observed_memory_percentile"]
        v1 = bool(obs_abs > pct95)
        row.update({
            "std_final": std, "abs_r": obs_abs, "r": pr["observed_memory_r"],
            "null_pct": null_pct, "V1": v1,
            "provenance": f"{SRC_R2_EXP2} (memory arm) + {SRC_R2_NULL} "
                          f"(null pct {null_pct:.2f})",
            "v1_source": "archived G2 null (identical protocol)",
            "family": _r2_family(base_dir),
        })
    elif rung == "R4":
        p2 = _load_json(os.path.join(base_dir, SRC_R4))["p2"]
        std = p2["spread_std"]
        early = p2["early_corr"]
        n_ord = p2.get("n_orderings")
        row.update({
            "substrate": "attachment",
            "std_final": std, "abs_r": abs(early), "r": early,
            "null_pct": None,
            # V1 imported from G1.5's established path-dependence; there is NO
            # identical-construction null for R4 (different axis/protocol).
            "V1": True,
            "provenance": (f"{SRC_R4} -> p2 (spread_std={std:.4f}, "
                           f"early_corr={early:+.4f}, n_orderings={n_ord}, "
                           f"attachment axis, field-simulated)"),
            "v1_source": "IMPORTED from G1.5 established path-dependence "
                         "(different substrate; no identical-null percentile)",
            "protocol_diff": (
                "G1.5 protocol DIFFERS from G3 Exp-2: attachment axis [-1,+1] "
                "(not caution [0,1]); 114-event multiset x 40 orderings (not "
                "40-event x 12); field-simulated (not real-model). std is the "
                "spread of final position across orderings in BOTH -- the "
                "comparable path-dependence metric -- but absolute levels are "
                "NOT comparable across substrates (fair-comparison note, "
                "prereg)."),
        })
    else:  # R1 / R3 (new)
        null = _load_json(os.path.join(base_dir, NULL_JSON_TMPL.format(rung=rung)))
        if null is None:
            row.update({
                "std_final": float("nan"), "abs_r": float("nan"),
                "r": float("nan"), "null_pct": None, "V1": None,
                "provenance": f"{NULL_JSON_TMPL.format(rung=rung)} (ABSENT -- "
                              f"run --exp2 {rung} then --null {rung})",
                "v1_source": "MISSING",
            })
        else:
            row.update({
                "std_final": null["memory_std_final"],
                "abs_r": null["observed_memory_abs_r"],
                "r": null["observed_memory_r"],
                "null_pct": null["observed_memory_percentile"],
                "V1": bool(null["V1_path_dependence_beyond_null"]),
                "provenance": f"{NULL_JSON_TMPL.format(rung=rung)} "
                              f"(own-null, identical G3 protocol)",
                "v1_source": "own random-walk null (identical protocol)",
            })
    return row


def _r2_family(base_dir):
    """Secondary R2 family replicates (G2.5 llama31 + ablit): std + null pct."""
    fam = {}
    for slug, (exp2_src, null_src) in SRC_R2_FAMILY.items():
        null = _load_json(os.path.join(base_dir, null_src))
        if null is None:
            continue
        fam[slug] = {
            "std_final": null.get("memory_std_final"),
            "abs_r": null.get("observed_memory_abs_r"),
            "null_pct": null.get("observed_memory_percentile"),
            "V1": bool(null.get("V1_path_dependence_beyond_null")),
            "provenance": null_src,
        }
    return fam


def _arm_stats_from(results, arm):
    """exp2_arm_stats but tolerant of a None results dict."""
    if results is None:
        return [], [], float("nan"), float("nan")
    return exp2_arm_stats(results, arm)


# ---- the frozen interpretation rule (classify) --------------------------
# The three property axes, frozen order for the monotonicity check.
AXES = ("compression", "recurrence", "coupling")

# Ordinal codings per axis (frozen). Higher = "more of the property". Used ONLY
# for the gradient (monotonicity) test. Coupling is the pre-registered
# hypothesis axis.
COUPLING_ORD = {"none": 0, "weak/implicit": 1, "STRONG": 2}
COMPRESSION_ORD = {  # low-D bounded = high; unbounded growing text = low
    "none (grows)": 0, "medium": 1, "high (scalar)": 2, None: 0,
}
RECURRENCE_ORD = {  # append-only = low; re-read+re-write = high
    "none": 0, "append-only": 1, "append + top-k retrieve": 1,
    "yes (self-rewrite)": 2, "yes": 2,
}
AXIS_ORD = {"compression": COMPRESSION_ORD, "recurrence": RECURRENCE_ORD,
            "coupling": COUPLING_ORD}

# The coupled set = rungs classified perception-coupling STRONG (prereg: R4).
COUPLED_COUPLING_LEVELS = {"STRONG"}


def classify(rows):
    """Apply the FROZEN interpretation rule (prereg 1.c) MECHANICALLY.

    Precedence (flagged interpretation, mirrors g25's documented F-precedence):
      WALL -> GRADIENT -> NEITHER.
    The two branches are NOT mutually exclusive (only-R4-passes also reads as a
    jump along the coupling axis), so a precedence is committed: WALL is checked
    first because it is the strong, specific claim; if it does not fire cleanly,
    GRADIENT is checked; otherwise NEITHER.

    WALL fires iff the set of rungs passing V1 is EXACTLY the coupled set:
      {rungs with coupling in {STRONG}} pass V1 AND all non-coupled rungs fail.
      (Rungs whose V1 is None/unknown block a clean wall -> not WALL.)

    GRADIENT fires iff V1 (or, as a fallback signal, the null percentile) rises
    MONOTONICALLY along one of the three property axes across the rungs whose V1
    is known. The percentile is the cross-rung comparable signal (each rung vs
    its OWN null); std is NOT used for the axis test because R4 is a different
    substrate (attachment vs caution) -- see the fair-comparison note.

    Returns a dict {branch, axis, detail, precedence}.
    """
    precedence = "WALL -> GRADIENT -> NEITHER (flagged interpretation; the two "
    precedence += ("branches are not mutually exclusive, so WALL -- the strong "
                   "specific claim -- is evaluated first; see classify() "
                   "docstring / g25 F-precedence analogue).")

    known = [r for r in rows if r["V1"] is not None]
    unknown = [r["rung"] for r in rows if r["V1"] is None]

    passing = {r["rung"] for r in rows if r["V1"] is True}
    coupled = {r["rung"] for r in rows
               if r.get("coupling") in COUPLED_COUPLING_LEVELS}
    non_coupled = {r["rung"] for r in rows
                   if r.get("coupling") not in COUPLED_COUPLING_LEVELS}

    # --- WALL --------------------------------------------------------------
    # Clean wall requires: no unknown V1 anywhere (an unknown could break it),
    # every coupled rung passes, every non-coupled rung fails.
    wall = (not unknown
            and coupled == passing
            and passing.issubset(coupled)
            and non_coupled.isdisjoint(passing))
    if wall:
        return {
            "branch": "WALL",
            "axis": "perception-coupling",
            "detail": {
                "passing": sorted(passing), "coupled": sorted(coupled),
                "conclusion": (
                    "attractor formation requires perception-coupling; memory "
                    "persistence and sophistication are insufficient (the "
                    "strong spine)."),
            },
            "precedence": precedence,
        }

    # --- GRADIENT ----------------------------------------------------------
    # Monotonic rise of the pass/percentile signal along one property axis.
    # Signal per rung: null_pct when defined, else a V1-derived proxy so R0
    # (control, pct None) and R4 (imported PASS, pct None) still slot on the
    # axis. We test monotonic non-decreasing with a strict rise somewhere.
    gradient_axis, gradient_detail = _gradient_test(known)
    if gradient_axis is not None:
        return {
            "branch": "GRADIENT",
            "axis": gradient_axis,
            "detail": gradient_detail,
            "precedence": precedence,
        }

    # --- NEITHER -----------------------------------------------------------
    return {
        "branch": "NEITHER",
        "axis": None,
        "detail": {
            "passing": sorted(passing),
            "coupled": sorted(coupled),
            "unknown_V1": unknown,
            "note": ("neither a clean perception-coupling wall nor a monotonic "
                     "gradient along a single property axis; reported honestly "
                     "per the prereg (no forcing)."),
        },
        "precedence": precedence,
    }


def _rung_signal(row):
    """Cross-rung comparable path-dependence signal for the gradient test.

    Signal = the null percentile (each rung vs its OWN null -- the fair
    cross-substrate signal), BUT a rung that FAILS V1 is FLOORED to 0. A
    high-but-failing percentile (e.g. R2 at 78) is NOT path-dependence -- it did
    not clear its own null -- so it must not feed a "path-dependence rises"
    monotonicity claim (advisor catch: otherwise a failing 78th-pct rung
    manufactures a false gradient). A PASS uses its real percentile if defined,
    else a V1-derived ceiling (R4 imported PASS). A None/unknown V1 is treated
    as the floor here (the caller excludes unknown rows anyway).
    """
    if row["V1"] is not True:
        # FAIL or unknown -> not path-dependent -> floor
        return 0.0
    pct = row.get("null_pct")
    if pct is not None and np.isfinite(pct):
        return float(pct)
    # PASS but no own-null percentile (R4 imported): ceiling
    return 100.0


def _axis_ordinal_vector(rows, axis):
    """Ordinal codings for `axis` across `rows` (in row order). None if any row
    is uncodable on that axis."""
    ordmap = AXIS_ORD[axis]
    vec = []
    for r in rows:
        lv = ordmap.get(r.get(axis))
        if lv is None:
            return None
        vec.append(lv)
    return vec


def _gradient_test(known_rows):
    """Return (axis_name, detail) if the path-dependence signal rises
    monotonically along AT LEAST ONE property axis; else (None, {}).

    Collects ALL monotone axes (not the first) and reports collinearity. On the
    frozen rung set compression and coupling share the identical ordinal vector
    [0,0,1,0,2] -- they are perfectly collinear and CANNOT be separated by the
    monotonicity test; any signal monotone along one is identically monotone
    along the other. Recurrence [0,1,2,1,2] is the only independently-resolvable
    axis. So the report:
      * names ALL monotone axes,
      * if the pre-registered COUPLING axis is among them, names it as the
        prereg-hypothesis axis,
      * flags any collinear pair (identical ordinal vectors on this rung set) as
        inseparable here -- the honest statement (the prereg's "three
        independent properties" design exists to prevent misattribution; when
        two axes are collinear on the tested rungs we say so rather than pick
        one).

    Ties in an axis ordinal collapse to the group MEAN signal before the
    monotonicity check (so several rungs at one level do not spuriously break
    monotonicity by their internal noise ordering).
    """
    monotone = []
    vectors = {}
    for axis in AXES:
        buckets = {}
        ok = True
        for r in known_rows:
            level = AXIS_ORD[axis].get(r.get(axis))
            if level is None:
                ok = False
                break
            buckets.setdefault(level, []).append(_rung_signal(r))
        if not ok or len(buckets) < 2:
            continue
        levels = sorted(buckets)
        means = [float(np.mean(buckets[lv])) for lv in levels]
        non_dec = all(means[i] <= means[i + 1] + 1e-9
                      for i in range(len(means) - 1))
        strict = any(means[i] + 1e-9 < means[i + 1]
                     for i in range(len(means) - 1))
        if non_dec and strict:
            monotone.append(axis)
            vectors[axis] = {"levels": levels, "mean_signal_per_level": means}

    if not monotone:
        return None, {}

    # collinearity: which monotone axes share an identical ordinal vector across
    # the known rungs (inseparable on this rung set).
    collinear_groups = []
    coded = {ax: _axis_ordinal_vector(known_rows, ax) for ax in monotone}
    seen = set()
    for i, ax in enumerate(monotone):
        if ax in seen:
            continue
        group = [ax]
        for other in monotone[i + 1:]:
            if coded[ax] is not None and coded[ax] == coded[other]:
                group.append(other)
                seen.add(other)
        seen.add(ax)
        if len(group) > 1:
            collinear_groups.append(group)

    # the named axis: prefer the pre-registered coupling hypothesis if monotone.
    named_axis = "coupling" if "coupling" in monotone else monotone[0]

    collinear_note = ""
    if collinear_groups:
        parts = ["{" + ", ".join(g) + "}" for g in collinear_groups]
        collinear_note = (
            "COLLINEARITY: on the frozen rung set the axes " + " and ".join(parts)
            + " have identical ordinal vectors and are INSEPARABLE by the "
            "monotonicity test -- a signal monotone along one is identically "
            "monotone along the other. Recurrence [0,1,2,1,2] is the only "
            "independently-resolvable axis. The prereg hypothesis is coupling; "
            "this ladder CANNOT distinguish coupling from a collinear partner "
            "(compression) on these rungs.")

    detail = {
        "axis": named_axis,
        "monotone_axes": monotone,
        "per_axis": vectors,
        "collinear_groups": collinear_groups,
        "collinearity_note": collinear_note,
        "signal": ("null percentile (own-null; V1-FAIL floored to 0; "
                   "V1-derived ceiling where pct undefined)"),
        "conclusion": (
            f"path-dependence rises monotonically along {monotone} "
            f"(named axis: {named_axis}). "
            + (collinear_note + " " if collinear_note else "")
            + "Report the gradient and name the axis/axes (prereg: the BETTER "
            "result -- a mechanistic contribution; do not massage into a wall)."),
    }
    return named_axis, detail


def _unfloored_axis_vectors(rows):
    """Informational: for each property axis, the RAW null percentile of each
    rung grouped by axis ordinal level -- WITHOUT the V1-fail floor the branch
    logic applies. This surfaces a sub-threshold rising trend (e.g. R2 at 78th
    pct while still failing V1) that the branch logic deliberately floors to 0,
    so a nascent gradient is visible to the supervisor without changing the
    authoritative wall/gradient/neither verdict. std is NOT used here (R4 is a
    different substrate); percentile is the cross-rung comparable signal.
    """
    out = {}
    for axis in AXES:
        ordmap = AXIS_ORD[axis]
        by_level = {}
        for r in rows:
            level = ordmap.get(r.get(axis))
            pct = r.get("null_pct")
            # imported/None-pct rungs: annotate with V1 so the row is not lost
            entry = {"rung": r["rung"],
                     "null_pct": (float(pct) if pct is not None
                                  and np.isfinite(pct) else None),
                     "V1": r["V1"]}
            by_level.setdefault(str(level), []).append(entry)
        out[axis] = by_level
    return out


def _grade_predictions(rows):
    """Grade the frozen per-rung V1 predictions (prereg table), no relabel.

    Committed V1: R0 FAIL, R1 FAIL, R2 FAIL, R3 FAIL, R4 PASS.
    """
    committed = {"R0": False, "R1": False, "R2": False, "R3": False,
                 "R4": True}
    grades = {}
    for r in rows:
        rung = r["rung"]
        exp = committed.get(rung)
        obs = r["V1"]
        if obs is None:
            verdict = "PENDING (V1 unknown -- run --exp2/--null)"
            ok = None
        else:
            ok = (obs == exp)
            verdict = "PASS" if ok else "FAIL(prediction wrong)"
        grades[rung] = {"committed_V1": exp, "observed_V1": obs,
                        "prediction_correct": ok, "verdict": verdict}
    return grades


def run_ladder(base_dir="."):
    """Assemble g3_ladder.json + print the ladder table; apply the frozen
    interpretation rule; grade the frozen predictions; produce the figure."""
    order = ["R0", "R1", "R2", "R3", "R4"]
    rows = [_rung_stats(r, base_dir) for r in order]

    # Operational completeness guard (NOT a scientific branch -- the prereg
    # froze exactly three outcomes wall/gradient/neither). If any rung's V1 is
    # unknown (a NEW rung not yet run), we print the diagnostic table but
    # WITHHOLD the authoritative interpretation: a pending rung could break a
    # wall or a gradient (e.g. a PASS on a none-coupling rung). The frozen run
    # order (gates -> exp2 -> ladder) guarantees R1/R3 are done first, so this
    # is a safety net, not a blocker. Mirrors run_null's missing-input refusal.
    pending = [r["rung"] for r in rows if r["V1"] is None]

    result = classify(rows)
    grades = _grade_predictions(rows)

    # --- print the ladder table -------------------------------------------
    print("=" * 96)
    print("G3 LADDER -- path-dependence across the memory architecture family")
    print("=" * 96)
    hdr = (f"  {'rung':4s} {'name':30s} {'coupling':13s} {'std':>7s} "
           f"{'|r|':>6s} {'nullpct':>8s} {'V1':>5s}  provenance")
    print(hdr)
    print("  " + "-" * 92)
    for r in rows:
        std = _fmt(r["std_final"])
        absr = _fmt(r["abs_r"])
        pct = _fmt(r["null_pct"]) if r["null_pct"] is not None else "  --"
        v1 = ("PASS" if r["V1"] is True else
              "FAIL" if r["V1"] is False else "  ??")
        sub = "" if r["substrate"] == "caution" else f" [{r['substrate']}]"
        name = (r["name"][:28] + sub)[:30]
        print(f"  {r['rung']:4s} {name:30s} {str(r['coupling']):13s} "
              f"{std:>7s} {absr:>6s} {pct:>8s} {v1:>5s}")
        print(f"       provenance: {r['provenance']}")
        if r.get("protocol_diff"):
            print(f"       PROTOCOL-DIFF FLAG: {r['protocol_diff']}")
        if r.get("family"):
            for slug, f in r["family"].items():
                print(f"       family[{slug}]: std={_fmt(f['std_final'])} "
                      f"nullpct={_fmt(f['null_pct'])} "
                      f"V1={'PASS' if f['V1'] else 'FAIL'} ({f['provenance']})")
    print("  " + "-" * 92)

    # --- frozen predictions -----------------------------------------------
    print("\n  FROZEN PER-RUNG PREDICTIONS (prereg, no relabel):")
    for r in order:
        g = grades[r]
        cexp = ("PASS" if g["committed_V1"] else "FAIL"
                if g["committed_V1"] is not None else "??")
        print(f"    {r}: committed V1={cexp:4s}  observed="
              f"{'PASS' if g['observed_V1'] is True else 'FAIL' if g['observed_V1'] is False else '??':4s}"
              f"  -> {g['verdict']}")

    # --- interpretation rule ----------------------------------------------
    print("\n" + "=" * 96)
    print("FROZEN INTERPRETATION RULE (prereg 1.c)")
    print("=" * 96)
    if pending:
        print(f"  *** INTERPRETATION WITHHELD: V1 unknown for {pending} "
              f"(NEW rung(s) not yet run). ***")
        print("  The diagnostic table above is complete; the authoritative "
              "wall/gradient/neither verdict is withheld until every rung's V1 "
              "is known (a pending rung could break a wall or a gradient). Run "
              "--exp2/--null for the pending rung(s), then re-run --ladder.")
        print(f"  (provisional branch on the KNOWN rungs, NOT authoritative: "
              f"{result['branch']}"
              + (f" / axis {result['axis']}" if result['axis'] else "") + ")")
    else:
        print(f"  precedence: {result['precedence']}")
        print(f"\n  >>> BRANCH: {result['branch']} <<<")
        if result["axis"]:
            print(f"  axis: {result['axis']}")
        det = result["detail"]
        if det.get("collinearity_note"):
            print(f"  {det['collinearity_note']}")
        if "conclusion" in det:
            print(f"  conclusion: {det['conclusion']}")
        elif "note" in det:
            print(f"  note: {det['note']}")
    print("\n  Honest ceiling (prereg, restated): a ladder is EVIDENCE FOR "
          "NECESSITY, never proof -- we cannot test every architecture.")
    print("  Fair-comparison note (prereg): R4's axis is attachment, the "
          "natural rungs' is caution; the comparable metric is the SPREAD of "
          "final position across orderings, never absolute levels across "
          "substrates. R4's V1 is imported from G1.5, not an identical null.")
    print("=" * 96)

    out = {
        "rungs": rows,
        "predictions": grades,
        "interpretation": result,
        "interpretation_withheld": bool(pending),
        "pending_rungs": pending,
        "axes_ordinals": {
            "compression": COMPRESSION_ORD, "recurrence": RECURRENCE_ORD,
            "coupling": COUPLING_ORD},
        # Informational: the UNFLOORED percentile-vs-axis vectors, so a
        # sub-threshold "rising but not-yet-significant" trend (which the
        # V1-fail floor deliberately hides from the branch logic, to prevent a
        # failing 78th-pct rung manufacturing a false gradient) is still visible
        # to the supervisor. Prereg: "be ready for a gradient, do not force a
        # wall" -- this surfaces the raw trend without letting it change the
        # authoritative branch.
        "unfloored_percentile_by_axis": _unfloored_axis_vectors(rows),
        "notes": {
            "fair_comparison": (
                "R4 is a different substrate (attachment axis, field-simulated, "
                "114-event x 40 orderings); its std is not directly comparable "
                "to the caution-axis rungs and its V1 is imported from G1.5's "
                "established path-dependence, not an identical-protocol null."),
            "gradient_signal": (
                "the cross-rung comparable path-dependence signal is the null "
                "percentile (each rung vs its OWN null); std is confined to the "
                "caution-substrate rungs and not used in the axis monotonicity "
                "test."),
        },
    }
    _atomic_write(os.path.join(base_dir, LADDER_JSON), out)
    print(f"written: {os.path.join(base_dir, LADDER_JSON)}")

    _make_ladder_figure(rows, result, base_dir)
    return out


def _make_ladder_figure(rows, result, base_dir):
    """g3_fig_ladder.png: std(final) per rung ordered along the
    perception-coupling axis, null band shaded, R4 marked different-substrate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # order rungs along the perception-coupling axis (ordinal), then by name.
    def coup_ord(r):
        return COUPLING_ORD.get(r.get("coupling"), -1)
    ordered = sorted(rows, key=lambda r: (coup_ord(r), r["rung"]))

    labels = [f"{r['rung']}\n{str(r['coupling'])}" for r in ordered]
    stds = [r["std_final"] if r["std_final"] is not None else np.nan
            for r in ordered]
    is_r4 = [r["rung"] == "R4" for r in ordered]
    v1 = [r["V1"] for r in ordered]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(ordered))

    # null band (caution-substrate noise floor): the prereg band 0.03-0.09 that
    # the archived caution rungs sit within. Shaded across the caution rungs.
    caution_x = [i for i, r in enumerate(ordered) if r["substrate"] == "caution"]
    if caution_x:
        ax.axhspan(0.03, 0.09, color="#bbbbbb", alpha=0.35, zorder=0,
                   label="caution-axis noise band (0.03-0.09)")

    colors = []
    for r, passed in zip(ordered, v1):
        if r["rung"] == "R4":
            colors.append("#8e44ad")            # different substrate
        elif passed is True:
            colors.append("#d1495b")            # V1 PASS
        elif passed is False:
            colors.append("#00798c")            # V1 FAIL
        else:
            colors.append("#cccccc")            # unknown

    bars = ax.bar(x, stds, color=colors, width=0.62, zorder=3)

    # mark V1 PASS/FAIL and R4-different-substrate
    for i, (r, b) in enumerate(zip(ordered, bars)):
        h = b.get_height()
        if np.isnan(h):
            ax.text(i, 0.01, "no data", ha="center", va="bottom", fontsize=8,
                    rotation=90, color="#888")
            continue
        tag = ("PASS" if r["V1"] is True else "FAIL" if r["V1"] is False
               else "??")
        ax.text(i, h + 0.012, tag, ha="center", va="bottom", fontsize=8,
                fontweight="bold",
                color="#8e44ad" if r["rung"] == "R4" else "#333")
        if r["rung"] == "R4":
            ax.text(i, h / 2, "different\nsubstrate\n(attachment)", ha="center",
                    va="center", fontsize=7.5, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("std(final) across the 12 orderings\n(spread of final "
                  "position = path-dependence)")
    ax.set_xlabel("rungs ordered along the perception-coupling axis "
                  "(none -> weak/implicit -> STRONG)")
    branch = result["branch"]
    ax.set_title(f"G3 memory ladder -- path-dependence vs perception-coupling\n"
                 f"frozen interpretation: BRANCH {branch}"
                 + (f" (axis: {result['axis']})" if result["axis"] else ""))
    ax.legend(fontsize=8, loc="upper left")

    # fair-comparison caption
    cap = ("R4 (purple) is a DIFFERENT SUBSTRATE: attachment axis [-1,+1], "
           "field-simulated, 114-event x 40 orderings -- its std is NOT "
           "directly comparable to the caution-axis rungs (R0-R3) and its V1 "
           "is imported from G1.5's established path-dependence, not an "
           "identical-protocol null. The comparable path-dependence signal "
           "across substrates is each rung's own-null percentile, not raw std.")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=7.2,
             wrap=True, color="#444")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(base_dir, LADDER_FIG), dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"written: {os.path.join(base_dir, LADDER_FIG)}")


# ==========================================================================
# small helpers
# ==========================================================================
def _probe_mean_at(probes, turn):
    for p in probes:
        if p["turn"] == turn:
            return p["mean"]
    return None


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x, sign=False):
    if x is None:
        return "  --"
    try:
        if not np.isfinite(x):
            return " nan"
    except TypeError:
        return str(x)
    return f"{x:+.3f}" if sign else f"{x:.3f}"


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="G3 ladder (subagent B)")
    ap.add_argument("--exp2", metavar="RUNG",
                    help="run the path-dependence exp2 for a NEW rung (R1|R3)")
    ap.add_argument("--null", metavar="RUNG",
                    help="run the per-rung random-walk null (R1|R3)")
    ap.add_argument("--ladder", action="store_true",
                    help="assemble the ladder table + figure + interpretation")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stub", action="store_true",
                    help="deterministic dry rehearsal (no daemon, no memory_g3)")
    args = ap.parse_args(argv)

    n_modes = sum(bool(x) for x in (args.exp2, args.null, args.ladder))
    if n_modes == 0:
        ap.error("pick one of --exp2 RUNG / --null RUNG / --ladder")
    if n_modes > 1:
        ap.error("run --exp2 / --null / --ladder separately")

    if args.exp2:
        run_exp2(args.exp2, args.stub, args.resume)
    elif args.null:
        run_null(args.null)
    elif args.ladder:
        run_ladder()
    return 0


if __name__ == "__main__":
    sys.exit(main())
