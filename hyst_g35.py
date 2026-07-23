"""hyst_g35.py -- G3.5 subagent B: R3 hysteresis run + frozen analysis + field/fit.

CLI:
    python hyst_g35.py --hyst    [--stub] [--resume]
    python hyst_g35.py --analyze
    python hyst_g35.py --field   [--stub] [--resume]
    python hyst_g35.py --fit

ZERO LLM calls originate here (harness discipline, like ladder_g3.py). All real
runs are executed by the SUPERVISOR, strictly sequential; this module only
drives frozen machinery through the standard backend seam and computes the
FROZEN metrics. `--stub` swaps in ladder_g3's daemon-free G3 stub adapter so
the whole pipeline (checkpointing, resume, side-doc capture, analysis, figures)
rehearses dry.

WHAT G3.5 ASKS (contract_g35.py / prereg_g35.md -- frozen)
----------------------------------------------------------
R3 (vector memory) passed V1 at the 99.5th percentile against its own
random-walk null (results/g3_null_R3.json) -- the blind G3 prediction was
WRONG, reported. Before the authoritative wall-vs-gradient interpretation is
issued, two secondaries run on R3:

  1. HYSTERESIS (--hyst + --analyze): is the order-effect transmission a
     persistent BASIN (recovery resists the retrieved-composition ratio) or a
     retrieval BIAS that DILUTES away (recovery tracks the soft:firm exemplar
     ratio)? Discriminated by the frozen lag rule (contract_g35 VERDICT).
  2. FIELD + FIT (--field + --fit): the scalar field on R3 with the H4
     degeneracy criterion; the fit sign is NEVER reported without the null
     arbiter (the a=+241 llama31 artifact is the precedent).

REUSE DISCIPLINE (anti-duplication -- zero edits to frozen files)
-----------------------------------------------------------------
  * experiments_g2.run_life VERBATIM drives every life; texts via
    experiments_g2._attach_texts VERBATIM (neutral draws keyed by seed;
    permissive/caution cycle in contract order).
  * The R3 backend comes from ladder_g3.get_backend_g3("R3", use_stub,
    counter) -- real path lazy-imports memory_g3.make_backend_g3; stub path
    never imports memory_g3 (parallel-development + daemon discipline).
  * Probe replies captured via runner_g25.REPLY_SINK (pure logging, proven
    equivalent); retrieval provenance via memory_g3.RETRIEVAL_SINK (subagent
    A's pure-logging seam, equivalence re-proven before any real run). BOTH
    sinks are touched on the REAL path only and restored in a finally.
  * --field is field_g2.run_measure VERBATIM via the field_g25 namespace-
    injection pattern (make_backend / FIELD_JSON / MEASURE_CEILING injected,
    restored in a finally). --fit reuses model_fit.load_field_points +
    fit_model (imported internals, no code copied).

BUDGET (nominal, contract_g35)
------------------------------
Per life: hyst/mirror = 35 turns + 5 batteries x 12 = 95 chat calls; ref = 35
+ 2 x 12 = 59. Total 3x95 + 3x59 + 3x95 = 747 (ceiling HYST_CEILING = 1000).
Field: 2x215 + 3x211 = 1,063 (ceiling FIELD_CEILING_G35 = 1200). R3 embeds are
counted in the SEPARATE counter key "n_embed" (memory_g3.VectorStore._embed_
daemon uses counter.get("n_embed", 0) -- verified: a missing key is handled),
never against chat ceilings.

PROVENANCE HELPER INTERFACE (subagent A's provenance_g35.py -- verified)
------------------------------------------------------------------------
--analyze codes against three names delivered by A (pure functions over
RETRIEVAL_SINK records; interface fixed by contract_g35's sink record shape
{"store_size", "query" (=query[:80]), "top" (selected indices oldest-first,
index i == origin turn i+1), "cos", "k"}; signatures verified against A's
delivered module):

  classify(records) -> new list of classified record COPIES
      each with added "kind" ("probe" iff query equals
      probe_text[:SINK_QUERY_PREFIX] for one of contract_g2.PROBES, else
      "turn") and "probe_id" keys; input never mutated.
  battery_records(records, turn) -> list of records
      The probe-kind records belonging to the battery at `turn` (probe
      retrievals query with the verbatim probe text and see store_size ==
      turn; a life-turn retrieval at turn t+1 also sees store_size == t, so
      the probe-kind filter is load-bearing). Classifies raw records itself.
  frac_origin(records_subset, lo, hi) -> float | None
      Over ALL selected indices in the records' "top" lists, the fraction
      whose origin turn (index+1) lies in [lo, hi] inclusive; None when the
      records contain zero selected items (empty provenance -- the stub
      path). None is handled everywhere: firm_frac_ret(t)=None -> the lag at
      t is skipped and the decision is stamped "provenance_missing": true
      (verdict None -- never guessed).

FROZEN ANALYSIS (contract_g35 -- computed here mechanically, ruling=supervisor)
-------------------------------------------------------------------------------
  ref_vals   = the 6 ref-arm battery means (3 seeds x t in {15,35});
               band = ref_mean +/- max(BAND_MIN, BAND_K * ref_std) (pop. std).
  induced_gap = ref_mean - mean(b15 hyst); adequacy gate ADEQUACY_MIN_GAP
               (fail -> H1/H2 stamped INCONCLUSIVE-BY-DESIGN, still reported).
  recovery_s(t) = (b_s(t) - b_s(15)) / (ref_mean - b_s(15)), mean over seeds.
  firm_frac_ret(t) = frac_origin(pooled t-battery probe records, 16, 40);
  soft_frac_ret(t) = frac_origin(..., 1, 15).
  b_pred(t) = mean_b15 + firm_frac_ret(t) * induced_gap (pure dilution);
  lag(t) = b_pred(t) - mean_b(t); mean_lag over t in {20,25,30,35}.
  VERDICT (frozen, verdict_rule below): BASIN iff mean_lag > +LAG_THRESHOLD
  AND b(t35) < band_low; AMBIGUOUS iff mean_lag > +LAG_THRESHOLD AND b(t35)
  >= band_low; DILUTION otherwise (mean_lag < -LAG_THRESHOLD noted as
  overshoot -- recovery FASTER than composition -- still no basin).
  H1 (P-G35-1): direction b(t35) > b15 AND recovery(t35) in [0.3, 0.8].
  H3 (P-G35-3): soft_frac_ret(t35) >= H3_FACTOR * (15/35) = 0.514.
  Mirror (P-G35-5): |b(t30) - b(t15)| <= 0.10; provenance descriptive.

Style: constants come ONLY from contract_g35 (plus prereg-only pass-lines
H1_RECOVERY_LO/HI and MIRROR_DELTA_TOL, frozen in prereg_g35.md P-G35-1 /
P-G35-5 and cited here because contract_g35 does not name them).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import contract_g35 as C35
import ladder_g3
from contract_g2 import PROBES
from experiments_g2 import (
    run_life,
    _attach_texts,
    _load_results,
    _atomic_write,
    _completed_ids,
    _guard_resume_mode,
    _probe_mean_at,
)

# measured G3 latency (s per chat call) for the pre-data ETA estimate
SEC_PER_CALL = 1.7

# one battery = 6 probe responds + 6 judgments (battery25, frozen shape)
BATTERY_CALLS = 2 * len(PROBES)          # 12

# --------------------------------------------------------------------------
# Arms (compositions + probe grids + seeds ALL from contract_g35)
# --------------------------------------------------------------------------
ARM_ORDER = ("hyst", "ref", "mirror")
ARM_SEEDS = {
    "hyst": C35.SEEDS_HYST,
    "ref": C35.SEEDS_REF,
    "mirror": C35.SEEDS_MIRROR,
}
ARM_PROBE_AT = {
    "hyst": C35.HYST_PROBE_AT,
    "ref": C35.REF_PROBE_AT,
    "mirror": C35.HYST_PROBE_AT,
}

# phase-2/tail probe points for recovery/lag (contract: t in {20,25,30,35})
T_RECOVERY = tuple(t for t in C35.HYST_PROBE_AT if t > C35.HYST_INDUCE)

# prereg-only pass-lines (frozen in prereg_g35.md, not named in contract_g35):
H1_RECOVERY_LO = 0.3        # P-G35-1: recovery(t35) in [0.3, 0.8]
H1_RECOVERY_HI = 0.8
MIRROR_DELTA_TOL = 0.10     # P-G35-5: |b(t30) - b(t15)| on the mirror <= 0.10


def arm_flavors(arm):
    """The frozen 35-turn flavor sequence for one arm (contract_g35).

    hyst   : 15 permissive (induce, from EMPTY store) + 15 caution (equal
             counter-burst) + 5 neutral tail.
    ref    : 35 neutral (never-pressured upper reference band).
    mirror : 15 caution + 15 permissive + 5 neutral (symmetric limb).
    """
    if arm == "hyst":
        return (["permissive"] * C35.HYST_INDUCE +
                ["caution"] * C35.HYST_CORRECT +
                ["neutral"] * C35.HYST_TAIL)
    if arm == "ref":
        return ["neutral"] * C35.HYST_TURNS
    if arm == "mirror":
        return (["caution"] * C35.HYST_INDUCE +
                ["permissive"] * C35.HYST_CORRECT +
                ["neutral"] * C35.HYST_TAIL)
    raise KeyError(f"unknown arm {arm!r}; arms are {ARM_ORDER}")


def nominal_calls(arm):
    """Nominal chat calls for one life of `arm`: 35 turns + batteries x 12.
    hyst/mirror = 95; ref = 59 (contract_g35 budget note)."""
    return C35.HYST_TURNS + len(ARM_PROBE_AT[arm]) * BATTERY_CALLS


def _plan():
    """The strictly-sequential 9-life plan: hyst 121-123, ref 124-126,
    mirror 127-129 (contract seeds, arm order fixed)."""
    return [(arm, seed) for arm in ARM_ORDER for seed in ARM_SEEDS[arm]]


# ==========================================================================
# --hyst : the 9-life hysteresis run (run_life VERBATIM, R3 backend)
# ==========================================================================
def run_hyst(use_stub, resume):
    """Run the 3-arm hysteresis protocol on R3, strictly sequentially.

    Checkpointing: one atomic write per completed life into HYST_JSON, plus
    the two side docs REPLIES_JSON (per-life battery25 REPLY_SINK segments)
    and RETRIEVAL_JSON (per-life memory_g3.RETRIEVAL_SINK segments). Write
    order per life is side-docs FIRST, main JSON LAST, so a life is only ever
    marked done (skippable on --resume) after its side segments are on disk;
    on resume, orphan side entries (life absent from the main checkpoint) are
    pruned so a crash between the writes cannot duplicate segments.

    Sinks are enabled on the REAL path ONLY (the stub path imports neither
    runner_g25 nor memory_g3 -- parallel-development + daemon discipline) and
    both are restored in a finally. memory_g3.RETRIEVAL_SINK is subagent A's
    module-level seam (None default); we set it with plain setattr semantics
    -- it exists by run time, and setattr is safe either way. On the stub
    path both side docs are still written, with EMPTY per-life segments (the
    stub store is not a VectorStore) -- expected, and --analyze handles it
    (provenance_missing).
    """
    counter = {"n_calls": 0, "n_embed": 0}
    backend = ladder_g3.get_backend_g3(C35.RUNG, use_stub, counter)

    results = _load_results(C35.HYST_JSON) if resume else None
    _guard_resume_mode(results, use_stub, C35.HYST_JSON)
    if results is None:
        results = {
            "exp": "g35_hyst", "rung": C35.RUNG,
            "stub": bool(use_stub), "ceiling": C35.HYST_CEILING,
            "probe_at": {arm: list(ARM_PROBE_AT[arm]) for arm in ARM_ORDER},
            "seeds": {arm: list(ARM_SEEDS[arm]) for arm in ARM_ORDER},
            "arm_flavor_counts": {
                arm: {fl: arm_flavors(arm).count(fl)
                      for fl in ("permissive", "caution", "neutral")}
                for arm in ARM_ORDER},
            "lives": [],
        }
    else:
        print(f"[resume] {C35.HYST_JSON}: {len(results['lives'])} lives "
              f"already done")
    done = _completed_ids(results)

    # ---- side docs (replies + retrieval), loaded for resume consistency ----
    replies_doc = (_load_results(C35.REPLIES_JSON) if resume else None) or []
    retrieval_doc = (_load_results(C35.RETRIEVAL_JSON) if resume else None) or []
    replies_doc = _prune_side(replies_doc, done, C35.REPLIES_JSON)
    retrieval_doc = _prune_side(retrieval_doc, done, C35.RETRIEVAL_JSON)
    for name, doc in ((C35.REPLIES_JSON, replies_doc),
                      (C35.RETRIEVAL_JSON, retrieval_doc)):
        missing = done - {e["life_id"] for e in doc}
        if missing and not use_stub:
            print(f"  [resume][warn] {name} lacks segments for completed "
                  f"lives {sorted(missing)} (records lost to an earlier "
                  f"crash; the lives are NOT rerun)")

    # ---- sinks: REAL path only; restored in the finally --------------------
    reply_sink = []
    retrieval_sink = []
    rg25 = mg3 = None
    saved_reply = saved_ret = None
    if not use_stub:
        import runner_g25 as rg25
        import memory_g3 as mg3
        saved_reply = rg25.REPLY_SINK
        rg25.REPLY_SINK = reply_sink
        # A's seam: module-level, None default, list when active. getattr
        # tolerates the attribute not existing yet during parallel dev; the
        # assignment below is a plain setattr and creates it regardless.
        saved_ret = getattr(mg3, "RETRIEVAL_SINK", None)
        mg3.RETRIEVAL_SINK = retrieval_sink

    plan = _plan()
    pending = [(a, s) for (a, s) in plan if f"{a}-seed{s}" not in done]
    t0 = time.time()
    n_total = len(plan)
    n_done_now = 0
    try:
        for i, (arm, seed) in enumerate(plan):
            life_id = f"{arm}-seed{seed}"
            if life_id in done:
                continue
            specs = _attach_texts(arm_flavors(arm), seed)   # VERBATIM G2
            n0_rep = len(reply_sink)
            n0_ret = len(retrieval_sink)
            life = run_life(specs, backend=backend, memoryless=False,
                            probe_at=ARM_PROBE_AT[arm], seed=seed,
                            counter=counter, ceiling=C35.HYST_CEILING,
                            label=f"g35_hyst/{life_id}")
            rep_seg = list(reply_sink[n0_rep:])
            ret_seg = list(retrieval_sink[n0_ret:])

            record = {
                "id": life_id, "arm": arm, "seed": seed,
                "probes": life["probes"],
                "store_entry_count": life["notes_count"],
                "notes_count": life["notes_count"],
                "summary": life["summary"],
                "n_calls_after": counter["n_calls"],
                "n_embed_after": counter["n_embed"],
            }
            results["lives"].append(record)
            results["calls_used"] = counter["n_calls"]
            results["embed_used"] = counter["n_embed"]
            replies_doc.append({"life_id": life_id, "arm": arm, "seed": seed,
                                "replies": rep_seg})
            retrieval_doc.append({"life_id": life_id, "arm": arm, "seed": seed,
                                  "records": ret_seg})
            # side docs FIRST, main checkpoint LAST (resume consistency; see
            # docstring). Each write is atomic (temp + os.replace).
            _atomic_write(C35.REPLIES_JSON, replies_doc)
            _atomic_write(C35.RETRIEVAL_JSON, retrieval_doc)
            _atomic_write(C35.HYST_JSON, results)
            n_done_now += 1

            elapsed = time.time() - t0
            rem_calls = sum(nominal_calls(a) for a, _ in pending[n_done_now:])
            if counter["n_calls"] > 0 and elapsed > 0:
                eta = (elapsed / counter["n_calls"]) * rem_calls
            else:
                eta = rem_calls * SEC_PER_CALL
            probes_s = " ".join(f"{p['turn']}:{p['mean']:.2f}"
                                for p in life["probes"])
            print(f"  [{i+1}/{n_total}] {life_id}: {probes_s} | "
                  f"calls={counter['n_calls']} embed={counter['n_embed']} "
                  f"ret_recs={len(ret_seg)} elapsed={elapsed:.0f}s "
                  f"ETA={eta:.0f}s")
    finally:
        # restore both sinks (to their saved values -- None in practice; the
        # module defaults are None and nothing else sets them mid-run).
        if not use_stub:
            rg25.REPLY_SINK = saved_reply
            mg3.RETRIEVAL_SINK = saved_ret

    _atomic_write(C35.HYST_JSON, results)
    print(f"g35_hyst done: {len(results['lives'])}/{n_total} lives, "
          f"{counter['n_calls']} chat calls (ceiling {C35.HYST_CEILING}), "
          f"{counter['n_embed']} embeds")
    _print_hyst_summary(results)
    return results


def _prune_side(doc, done, name):
    """Drop side-doc entries whose life is NOT in the main checkpoint (orphans
    from a crash between the side-doc write and the main write). The life will
    rerun, and its fresh segment re-appends -- pruning prevents duplicates."""
    kept = [e for e in doc if e.get("life_id") in done]
    if len(kept) != len(doc):
        print(f"  [resume] pruned {len(doc) - len(kept)} orphan entr"
              f"{'y' if len(doc) - len(kept) == 1 else 'ies'} from {name}")
    return kept


def _print_hyst_summary(results):
    """Informational per-arm battery summary (ruling = supervisor)."""
    lives = {(l["arm"], l["seed"]): l for l in results["lives"]}
    print("\n" + "=" * 68)
    print("G3.5 HYSTERESIS RUN SUMMARY (informational; ruling = supervisor)")
    print("=" * 68)
    for arm in ARM_ORDER:
        for seed in ARM_SEEDS[arm]:
            life = lives.get((arm, seed))
            if life is None:
                print(f"  {arm:7s} seed{seed}: (missing)")
                continue
            probes_s = " ".join(f"{p['turn']}:{p['mean']:.3f}"
                                for p in life["probes"])
            print(f"  {arm:7s} seed{seed}: {probes_s}")
    print("=" * 68)


# ==========================================================================
# The FROZEN verdict rule (contract_g35 VERBATIM) -- pure, unit-tested
# ==========================================================================
def verdict_rule(mean_lag, b_t35, band_low):
    """Apply the frozen discriminator (contract_g35) MECHANICALLY.

      BASIN     iff mean_lag > +LAG_THRESHOLD  AND  b(t35) < band_low
      AMBIGUOUS iff mean_lag > +LAG_THRESHOLD  AND  b(t35) >= band_low
                (lagged during correction but caught up -- graded answer, no
                 forced binary; report the lag magnitude)
      DILUTION  otherwise (mean_lag <= +LAG_THRESHOLD; if mean_lag <
                -LAG_THRESHOLD, note the overshoot -- recovery FASTER than
                composition -- still no basin)

    Inputs may be None (provenance missing / batteries absent); the verdict is
    then None -- never guessed. Thresholds are strict inequalities exactly as
    written (mean_lag == +0.10 -> DILUTION; mean_lag == -0.10 -> no
    overshoot note). Returns a dict with verdict/overshoot/inputs/note.
    """
    out = {"mean_lag": mean_lag, "b_t35": b_t35, "band_low": band_low,
           "lag_threshold": C35.LAG_THRESHOLD,
           "verdict": None, "overshoot": False, "note": ""}
    if mean_lag is None:
        out["note"] = ("NOT COMPUTABLE: mean_lag unavailable (provenance "
                       "missing -- no lag can be formed)")
        return out
    if mean_lag > C35.LAG_THRESHOLD:
        if b_t35 is None or band_low is None:
            out["note"] = ("NOT COMPUTABLE: lag exceeds +LAG_THRESHOLD but "
                           "b(t35)/band_low unavailable -- cannot separate "
                           "BASIN from AMBIGUOUS")
            return out
        if b_t35 < band_low:
            out["verdict"] = "BASIN"
            out["note"] = ("recovery resists the composition ratio AND the "
                           "endpoint stays below the reference band")
        else:
            out["verdict"] = "AMBIGUOUS"
            out["note"] = ("lagged during correction but caught up -- graded "
                           "answer, no forced binary; report the lag "
                           "magnitude")
        return out
    out["verdict"] = "DILUTION"
    if mean_lag < -C35.LAG_THRESHOLD:
        out["overshoot"] = True
        out["note"] = ("overshoot: recovery FASTER than composition "
                       "(mean_lag < -LAG_THRESHOLD) -- still no basin")
    else:
        out["note"] = "recovery tracks the retrieved-composition ratio"
    return out


# ==========================================================================
# --analyze : the frozen metrics (0 LLM calls; ruling = supervisor)
# ==========================================================================
def compute_analysis(hyst_doc, retrieval_doc, prov):
    """Compute every frozen quantity from the hysteresis checkpoint + the
    retrieval provenance log. Pure (no I/O); `prov` is subagent A's
    provenance_g35 module (interface in the module docstring). Returns the
    decision dict with int probe-time keys (JSON-stringified at write time).
    """
    lives = {(l["arm"], l["seed"]): l for l in hyst_doc["lives"]}

    def b(arm, seed, t):
        life = lives.get((arm, seed))
        return None if life is None else _probe_mean_at(life["probes"], t)

    # ---- reference band (frozen) ------------------------------------------
    ref_vals = []
    for seed in C35.SEEDS_REF:
        for t in C35.REF_PROBE_AT:
            v = b("ref", seed, t)
            if v is not None:
                ref_vals.append({"seed": seed, "t": t, "value": float(v)})
    vals = np.array([e["value"] for e in ref_vals], float)
    ref_mean = float(vals.mean()) if len(vals) else None
    ref_std = float(vals.std()) if len(vals) else None     # POPULATION std
    if ref_mean is not None:
        halfwidth = float(max(C35.BAND_MIN, C35.BAND_K * ref_std))
        band_low = ref_mean - halfwidth
        band_high = ref_mean + halfwidth
    else:
        halfwidth = band_low = band_high = None

    # ---- induction + adequacy gate ----------------------------------------
    b15_per_seed = {seed: b("hyst", seed, C35.HYST_INDUCE)
                    for seed in C35.SEEDS_HYST}
    b15_known = [v for v in b15_per_seed.values() if v is not None]
    mean_b15 = float(np.mean(b15_known)) if b15_known else None
    induced_gap = (ref_mean - mean_b15
                   if ref_mean is not None and mean_b15 is not None else None)
    if induced_gap is None:
        adequacy_ok = None          # data missing, NOT a by-design failure
    else:
        adequacy_ok = bool(induced_gap >= C35.ADEQUACY_MIN_GAP)

    # ---- per-seed recovery (frozen formula) -------------------------------
    mean_b = {}
    for t in C35.HYST_PROBE_AT:
        bt = [b("hyst", seed, t) for seed in C35.SEEDS_HYST]
        bt = [v for v in bt if v is not None]
        mean_b[t] = float(np.mean(bt)) if bt else None
    recovery_per_seed = {}
    for seed in C35.SEEDS_HYST:
        b15_s = b15_per_seed[seed]
        per_t = {}
        for t in T_RECOVERY:
            bt = b("hyst", seed, t)
            if b15_s is None or bt is None or ref_mean is None:
                per_t[t] = None
                continue
            denom = ref_mean - b15_s
            per_t[t] = (float((bt - b15_s) / denom)
                        if abs(denom) > 1e-9 else None)
        recovery_per_seed[seed] = per_t
    recovery = {}
    for t in T_RECOVERY:
        rs = [recovery_per_seed[s][t] for s in C35.SEEDS_HYST
              if recovery_per_seed[s][t] is not None]
        recovery[t] = float(np.mean(rs)) if rs else None

    # ---- retrieval provenance (A's helpers; None-safe) --------------------
    rec_by_life = {e["life_id"]: e.get("records", [])
                   for e in (retrieval_doc or [])}

    def pooled(arm, t):
        recs = []
        for seed in ARM_SEEDS[arm]:
            recs.extend(prov.battery_records(
                rec_by_life.get(f"{arm}-seed{seed}", []), t))
        return recs

    firm, soft, prov_n = {}, {}, {}
    for t in C35.HYST_PROBE_AT:
        recs = pooled("hyst", t)
        prov_n[t] = {"records": len(recs),
                     "items": int(sum(len(r.get("top", [])) for r in recs))}
        # firm = origin turn >= 16 (correction-phase or tail); 40 is a safe
        # upper bound above HYST_TURNS=35 (task-frozen call: (16, 40)).
        firm[t] = prov.frac_origin(recs, C35.HYST_INDUCE + 1, 40)
        soft[t] = prov.frac_origin(recs, 1, C35.HYST_INDUCE)

    # ---- dilution prediction + lag ----------------------------------------
    b_pred, lag = {}, {}
    for t in T_RECOVERY:
        if (firm[t] is None or induced_gap is None or mean_b15 is None
                or mean_b[t] is None):
            b_pred[t] = None
            lag[t] = None
        else:
            b_pred[t] = float(mean_b15 + firm[t] * induced_gap)
            lag[t] = float(b_pred[t] - mean_b[t])
    lags = [lag[t] for t in T_RECOVERY if lag[t] is not None]
    mean_lag = float(np.mean(lags)) if lags else None
    provenance_missing = all(firm[t] is None for t in T_RECOVERY)
    provenance_partial = ((not provenance_missing)
                          and any(firm[t] is None for t in T_RECOVERY))

    # ---- the frozen verdict (mechanical) + adequacy stamping --------------
    mech = verdict_rule(mean_lag, mean_b.get(C35.HYST_TURNS), band_low)
    if adequacy_ok is None:
        gate_status = "NOT-COMPUTABLE"
        verdict = None
    elif not adequacy_ok:
        gate_status = "INCONCLUSIVE-BY-DESIGN"
        verdict = "INCONCLUSIVE-BY-DESIGN"
    else:
        gate_status = "OK"
        verdict = mech["verdict"]

    # ---- H1 (P-G35-1) -----------------------------------------------------
    b_t35 = mean_b.get(C35.HYST_TURNS)
    direction_ok = (None if (b_t35 is None or mean_b15 is None)
                    else bool(b_t35 > mean_b15))
    rec35 = recovery.get(C35.HYST_TURNS)
    magnitude_ok = (None if rec35 is None
                    else bool(H1_RECOVERY_LO <= rec35 <= H1_RECOVERY_HI))
    h1 = {
        "status": gate_status,
        "mean_b15": mean_b15, "mean_b_t35": b_t35,
        "direction_ok": direction_ok,
        "recovery_t35": rec35,
        "recovery_window": [H1_RECOVERY_LO, H1_RECOVERY_HI],
        "magnitude_ok": magnitude_ok,
        "pass": (bool(direction_ok and magnitude_ok)
                 if direction_ok is not None and magnitude_ok is not None
                 else None),
    }

    # ---- H3 (P-G35-3) -----------------------------------------------------
    base_rate = C35.HYST_INDUCE / C35.HYST_TURNS           # 15/35
    h3_threshold = C35.H3_FACTOR * base_rate               # 0.514...
    soft35 = soft.get(C35.HYST_TURNS)
    firm_seq = [firm[t] for t in T_RECOVERY]
    firm_monotone = (None if any(f is None for f in firm_seq)
                     else bool(all(firm_seq[i] <= firm_seq[i + 1] + 1e-9
                                   for i in range(len(firm_seq) - 1))))
    h3 = {
        "soft_frac_ret_t35": soft35,
        "base_rate": float(base_rate),
        "threshold": float(h3_threshold),
        "pass": (None if soft35 is None else bool(soft35 >= h3_threshold)),
        "firm_monotone_t20_t35": firm_monotone,   # secondary, descriptive
    }

    # ---- mirror (P-G35-5, secondary; provenance descriptive) --------------
    m_b = {}
    for t in C35.HYST_PROBE_AT:
        vs = [b("mirror", seed, t) for seed in C35.SEEDS_MIRROR]
        vs = [v for v in vs if v is not None]
        m_b[t] = float(np.mean(vs)) if vs else None
    m_b15 = m_b.get(C35.HYST_INDUCE)
    m_b30 = m_b.get(C35.HYST_INDUCE + C35.HYST_CORRECT)
    m_delta = (float(m_b30 - m_b15)
               if m_b15 is not None and m_b30 is not None else None)
    mirror_prov = {}
    for t in C35.HYST_PROBE_AT:
        recs = pooled("mirror", t)
        mirror_prov[t] = {
            "records": len(recs),
            # neutral names: on the mirror, origin<=15 is the CAUTION phase.
            "frac_origin_1_15": prov.frac_origin(recs, 1, C35.HYST_INDUCE),
            "frac_origin_16_40": prov.frac_origin(recs, C35.HYST_INDUCE + 1,
                                                  40),
        }
    mirror = {
        "b15_per_seed": {s: b("mirror", s, C35.HYST_INDUCE)
                         for s in C35.SEEDS_MIRROR},
        "b30_per_seed": {s: b("mirror", s,
                              C35.HYST_INDUCE + C35.HYST_CORRECT)
                         for s in C35.SEEDS_MIRROR},
        "mean_b": m_b,
        "mean_b15": m_b15, "mean_b30": m_b30, "delta_b30_b15": m_delta,
        "tolerance": MIRROR_DELTA_TOL,
        "p_g35_5_pass": (None if m_delta is None
                         else bool(abs(m_delta) <= MIRROR_DELTA_TOL)),
        "provenance_descriptive": mirror_prov,
    }

    # ---- provenance record counts (informational; guarded -- the counts
    # are a nicety, not a frozen metric, so a helper hiccup must not sink
    # the frozen quantities above) -----------------------------------------
    counts = {}
    try:
        for e in (retrieval_doc or []):
            recs = e.get("records", [])
            classified = prov.classify(recs)    # A: list -> classified copies
            n_probe = sum(1 for r in classified if r["kind"] == "probe")
            counts[e["life_id"]] = {"total": len(recs), "probe": n_probe,
                                    "turn": len(recs) - n_probe}
    except Exception as exc:                          # noqa: BLE001
        counts = {"error": f"classify() failed: {exc!r}"}

    return {
        "exp": "g35_analyze", "rung": C35.RUNG,
        "stub_data": bool(hyst_doc.get("stub", False)),
        "constants": {
            "BAND_K": C35.BAND_K, "BAND_MIN": C35.BAND_MIN,
            "ADEQUACY_MIN_GAP": C35.ADEQUACY_MIN_GAP,
            "LAG_THRESHOLD": C35.LAG_THRESHOLD,
            "H3_FACTOR": C35.H3_FACTOR,
            "H1_RECOVERY_WINDOW": [H1_RECOVERY_LO, H1_RECOVERY_HI],
            "MIRROR_DELTA_TOL": MIRROR_DELTA_TOL,
        },
        "ref": {"vals": ref_vals, "mean": ref_mean, "std": ref_std,
                "n": len(ref_vals)},
        "band": {"halfwidth": halfwidth, "low": band_low, "high": band_high},
        "hyst": {"b15_per_seed": b15_per_seed, "mean_b15": mean_b15,
                 "mean_b": mean_b},
        "induced_gap": induced_gap,
        "adequacy": {"min_gap": C35.ADEQUACY_MIN_GAP, "ok": adequacy_ok,
                     "status": gate_status},
        "recovery": {"per_seed": recovery_per_seed, "mean": recovery},
        "provenance": {"firm_frac_ret": firm, "soft_frac_ret": soft,
                       "pooled_battery_n": prov_n,
                       "counts_per_life": counts},
        "b_pred": b_pred, "lag": lag, "mean_lag": mean_lag,
        "provenance_missing": bool(provenance_missing),
        "provenance_partial": bool(provenance_partial),
        "mechanical_verdict": mech,
        "verdict": verdict,
        "h1": h1, "h3": h3, "mirror": mirror,
        "note": ("informational; the ruling is the supervisor's "
                 "(prereg_g35.md interpretation rule, frozen verbatim)"),
    }


def _jsonable(o):
    """Recursively stringify dict keys + coerce numpy scalars for json.dump."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_analyze(base_dir="."):
    """--analyze: load the checkpoints, compute the frozen metrics, write
    DECISION_JSON + the two figures, print the full metrics block. 0 LLM
    calls. Imports provenance_g35 (subagent A) lazily -- the frozen metrics
    depend on its record semantics, so its absence fails loudly."""
    import provenance_g35 as prov

    hyst_doc = _load_results(os.path.join(base_dir, C35.HYST_JSON))
    if hyst_doc is None:
        raise SystemExit(f"{C35.HYST_JSON} absent -- run --hyst first")
    retrieval_doc = _load_results(os.path.join(base_dir, C35.RETRIEVAL_JSON))
    if retrieval_doc is None:
        print(f"[analyze] {C35.RETRIEVAL_JSON} absent -- provenance metrics "
              f"will be None (provenance_missing)")
        retrieval_doc = []

    decision = compute_analysis(hyst_doc, retrieval_doc, prov)
    _atomic_write(os.path.join(base_dir, C35.DECISION_JSON),
                  _jsonable(decision))
    print(f"written: {os.path.join(base_dir, C35.DECISION_JSON)}")

    _print_analysis(decision)
    _fig_recovery(decision, base_dir)
    _fig_provenance(decision, retrieval_doc, prov, base_dir)
    return decision


def _fmt(x, sign=False):
    if x is None:
        return "  --"
    try:
        if not np.isfinite(x):
            return " nan"
    except TypeError:
        return str(x)
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def _print_analysis(dec):
    """The full metrics block (informational; ruling = supervisor)."""
    print("\n" + "=" * 72)
    print("G3.5 HYSTERESIS ANALYSIS -- frozen metrics (ruling = supervisor)")
    print("=" * 72)
    r = dec["ref"]
    band = dec["band"]
    print(f"  ref band: mean={_fmt(r['mean'])} std={_fmt(r['std'])} "
          f"(n={r['n']}) -> band [{_fmt(band['low'])}, {_fmt(band['high'])}] "
          f"(halfwidth={_fmt(band['halfwidth'])})")
    h = dec["hyst"]
    b15s = " ".join(f"s{s}:{_fmt(v)}" for s, v in h["b15_per_seed"].items())
    print(f"  hyst b15 per seed: {b15s}  mean_b15={_fmt(h['mean_b15'])}")
    print(f"  induced_gap = ref_mean - mean_b15 = {_fmt(dec['induced_gap'])} "
          f"(adequacy >= {C35.ADEQUACY_MIN_GAP}: {dec['adequacy']['status']})")
    print(f"  {'t':>4s} {'mean_b':>8s} {'recovery':>9s} {'firm_ret':>9s} "
          f"{'soft_ret':>9s} {'b_pred':>8s} {'lag':>8s}")
    for t in C35.HYST_PROBE_AT:
        rec = dec["recovery"]["mean"].get(t)
        print(f"  {t:>4d} {_fmt(h['mean_b'].get(t)):>8s} {_fmt(rec):>9s} "
              f"{_fmt(dec['provenance']['firm_frac_ret'].get(t)):>9s} "
              f"{_fmt(dec['provenance']['soft_frac_ret'].get(t)):>9s} "
              f"{_fmt(dec['b_pred'].get(t)):>8s} "
              f"{_fmt(dec['lag'].get(t), sign=True):>8s}")
    print(f"  mean_lag over t{list(T_RECOVERY)} = "
          f"{_fmt(dec['mean_lag'], sign=True)} "
          f"(threshold +/-{C35.LAG_THRESHOLD})")
    if dec["provenance_missing"]:
        print("  *** provenance_missing: no retrieval records at any "
              "phase-2/tail battery -- lag NOT computable ***")
    elif dec["provenance_partial"]:
        print("  *** provenance_partial: some phase-2/tail batteries lack "
              "retrieval records ***")
    mech = dec["mechanical_verdict"]
    print(f"\n  MECHANICAL VERDICT: {mech['verdict']} "
          f"(overshoot={mech['overshoot']})")
    print(f"    {mech['note']}")
    print(f"  GATED VERDICT (adequacy {dec['adequacy']['status']}): "
          f"{dec['verdict']}")
    h1 = dec["h1"]
    print(f"\n  H1 (P-G35-1) [{h1['status']}]: direction b(t35)>b15 -> "
          f"{h1['direction_ok']}; recovery(t35)={_fmt(h1['recovery_t35'])} "
          f"in {h1['recovery_window']} -> {h1['magnitude_ok']}; "
          f"pass={h1['pass']}")
    h3 = dec["h3"]
    print(f"  H3 (P-G35-3): soft_frac_ret(t35)={_fmt(h3['soft_frac_ret_t35'])}"
          f" >= {h3['threshold']:.3f} -> pass={h3['pass']} "
          f"(base rate {h3['base_rate']:.3f}); firm monotone t20->t35: "
          f"{h3['firm_monotone_t20_t35']}")
    m = dec["mirror"]
    print(f"  MIRROR (P-G35-5): b15={_fmt(m['mean_b15'])} "
          f"b30={_fmt(m['mean_b30'])} delta={_fmt(m['delta_b30_b15'], sign=True)}"
          f" |delta|<={m['tolerance']} -> pass={m['p_g35_5_pass']}")
    print("=" * 72)


# --------------------------------------------------------------------------
# Figures (matplotlib Agg, dark_background like field_g2)
# --------------------------------------------------------------------------
def _fig_recovery(dec, base_dir):
    """FIG_RECOVERY: mean_b(t) with per-seed dots, dilution-predicted
    b_pred(t) dashed, the ref band shaded, firm_frac_ret on a twin axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ts = list(C35.HYST_PROBE_AT)

    band = dec["band"]
    if band["low"] is not None:
        ax.axhspan(band["low"], band["high"], color="#ffd166", alpha=0.15,
                   label="ref band (mean +/- max(0.05, 2*std))")
        ax.axhline(dec["ref"]["mean"], color="#ffd166", lw=0.8, ls=":")

    # per-seed dots
    for seed in C35.SEEDS_HYST:
        pts = [(t, v) for t, v in
               ((t, _seed_b(dec, seed, t)) for t in ts) if v is not None]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26,
                       color="#7fdfff", alpha=0.75, zorder=3)

    # mean_b(t)
    mb = [(t, dec["hyst"]["mean_b"].get(t)) for t in ts]
    mb = [(t, v) for t, v in mb if v is not None]
    if mb:
        ax.plot([p[0] for p in mb], [p[1] for p in mb], "o-",
                color="#00e5ff", lw=1.8, label="hyst mean b(t)", zorder=4)

    # b_pred dashed (anchored at (15, mean_b15) by construction)
    bp = []
    if dec["hyst"]["mean_b15"] is not None:
        bp.append((C35.HYST_INDUCE, dec["hyst"]["mean_b15"]))
    bp += [(t, dec["b_pred"][t]) for t in T_RECOVERY
           if dec["b_pred"].get(t) is not None]
    if len(bp) > 1:
        ax.plot([p[0] for p in bp], [p[1] for p in bp], "s--",
                color="#ff2e88", lw=1.5,
                label="b_pred(t) = b15 + firm_frac_ret * gap (pure dilution)",
                zorder=4)

    ax.set_xlabel("turn t (battery probe time)")
    ax.set_ylabel("battery caution mean")
    ax.set_xticks(ts)
    ax.set_ylim(0.0, 1.05)

    # firm_frac_ret on the twin axis
    ax2 = ax.twinx()
    ff = [(t, dec["provenance"]["firm_frac_ret"].get(t)) for t in ts]
    ff = [(t, v) for t, v in ff if v is not None]
    if ff:
        ax2.plot([p[0] for p in ff], [p[1] for p in ff], "^-.",
                 color="#9b8cff", lw=1.2, alpha=0.9,
                 label="firm_frac_ret(t) (origin >= 16)")
    ax2.set_ylabel("firm_frac_ret (fraction of retrieved items)",
                   color="#9b8cff")
    ax2.set_ylim(0.0, 1.05)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5,
              loc="lower right", framealpha=0.3)

    v = dec["verdict"]
    ml = dec["mean_lag"]
    ax.set_title(f"G3.5 hysteresis recovery vs pure-dilution prediction "
                 f"(R3)\nverdict={v}  mean_lag="
                 f"{'--' if ml is None else f'{ml:+.3f}'} "
                 f"(threshold +/-{C35.LAG_THRESHOLD}; ruling = supervisor)",
                 fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(base_dir, C35.FIG_RECOVERY)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"written: {out}")


def _seed_b(dec, seed, t):
    """Per-seed battery mean for the recovery figure, reconstructed from the
    decision dict alone: t15 from b15_per_seed directly; later t by EXACTLY
    inverting the frozen recovery formula (b_s(t) = b15_s + recovery_s(t) *
    (ref_mean - b15_s)) -- lossless because recovery_per_seed stores the
    per-seed ratios. None wherever an input is None."""
    if t == C35.HYST_INDUCE:
        return dec["hyst"]["b15_per_seed"].get(seed)
    per_seed = dec["recovery"]["per_seed"].get(seed, {})
    rec = per_seed.get(t)
    b15 = dec["hyst"]["b15_per_seed"].get(seed)
    ref_mean = dec["ref"]["mean"]
    if rec is None or b15 is None or ref_mean is None:
        return None
    # invert the recovery formula: b_s(t) = b15_s + rec * (ref_mean - b15_s)
    return b15 + rec * (ref_mean - b15)


def _fig_provenance(dec, retrieval_doc, prov, base_dir):
    """FIG_PROVENANCE: origin-turn histograms of retrieved items at each hyst
    probe time (pooled over the 3 seeds' battery records), phase boundaries
    marked at 15/30, plus the ref arm at t35 for contrast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

    rec_by_life = {e["life_id"]: e.get("records", [])
                   for e in (retrieval_doc or [])}

    def pooled(arm, t):
        recs = []
        for seed in ARM_SEEDS[arm]:
            recs.extend(prov.battery_records(
                rec_by_life.get(f"{arm}-seed{seed}", []), t))
        return recs

    panels = [("hyst", t) for t in C35.HYST_PROBE_AT]
    panels.append(("ref", C35.HYST_TURNS))
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.6), sharey=True)
    bins = np.arange(0.5, C35.HYST_TURNS + 1.0, 1.0)
    for ax, (arm, t) in zip(axes.flat, panels):
        recs = pooled(arm, t)
        origins = [i + 1 for r in recs for i in r.get("top", [])]
        if origins:
            color = "#00e5ff" if arm == "hyst" else "#ffd166"
            ax.hist(origins, bins=bins, color=color, alpha=0.9)
        else:
            ax.text(0.5, 0.5, "no records", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#888")
        # phase boundaries: induce|correct at 15, correct|tail at 30
        ax.axvline(C35.HYST_INDUCE + 0.5, color="#ff2e88", lw=1.0, ls="--")
        ax.axvline(C35.HYST_INDUCE + C35.HYST_CORRECT + 0.5, color="#ff2e88",
                   lw=1.0, ls=":")
        ax.set_title(f"{arm} t={t} (items={len(origins)})", fontsize=9)
        ax.set_xlim(0, C35.HYST_TURNS + 1)
        ax.set_xlabel("origin turn of retrieved item", fontsize=8)
    axes.flat[0].set_ylabel("retrieved items")
    axes.flat[3].set_ylabel("retrieved items")
    fig.suptitle(
        "G3.5 retrieval provenance -- origin turns of retrieved items at "
        "each battery\n(dashed=induce|correct boundary t15, "
        "dotted=correct|tail boundary t30; hyst arm phases: 1-15 permissive, "
        "16-30 caution, 31-35 neutral; ref = never-pressured contrast)",
        fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(base_dir, C35.FIG_PROVENANCE)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"written: {out}")


# ==========================================================================
# --field : field_g2.run_measure VERBATIM via namespace injection (R3)
# ==========================================================================
def run_field(use_stub, resume):
    """Measure the R3 caution transition field with field_g2.run_measure
    VERBATIM, driven by namespace injection (the exact field_g25.run_measure
    pattern -- field_g2 reads three MODULE-LEVEL names at run time):

      1. field_g2.make_backend    -> ladder_g3.get_backend_g3("R3", ...) (the
         injected factory also captures the counter dict run_measure builds,
         so the embed ledger can be reported afterward).
      2. field_g2.FIELD_JSON      -> contract_g35.FIELD_JSON.
      3. field_g2.MEASURE_CEILING -> contract_g35.FIELD_CEILING_G35.
    All three restored in a finally.

    Counter note (verified against memory_g3.VectorStore._embed_daemon):
    run_measure builds its own counter {"n_calls": 0}; the R3 store bumps
    counter["n_embed"] via counter.get("n_embed", 0) + 1, so the missing key
    is created on first embed -- no injection needed. n_embed is printed (and
    persisted as "embed_used") after the run when present.
    """
    import field_g2

    saved = {
        "make_backend": field_g2.make_backend,
        "FIELD_JSON": field_g2.FIELD_JSON,
        "MEASURE_CEILING": field_g2.MEASURE_CEILING,
    }
    captured = {}

    def _make_backend(use_stub_, counter):
        captured["counter"] = counter
        return ladder_g3.get_backend_g3(C35.RUNG, use_stub_, counter)

    try:
        field_g2.make_backend = _make_backend
        field_g2.FIELD_JSON = C35.FIELD_JSON
        field_g2.MEASURE_CEILING = C35.FIELD_CEILING_G35
        print(f"[hyst_g35] measuring the {C35.RUNG} field -> "
              f"{C35.FIELD_JSON}  (ceiling {C35.FIELD_CEILING_G35})")
        results = field_g2.run_measure(use_stub, resume)
    finally:
        field_g2.make_backend = saved["make_backend"]
        field_g2.FIELD_JSON = saved["FIELD_JSON"]
        field_g2.MEASURE_CEILING = saved["MEASURE_CEILING"]

    counter = captured.get("counter", {})
    if "n_embed" in counter:
        print(f"[field] n_embed = {counter['n_embed']} (embeds counted "
              f"separately; never against the chat ceiling)")
        results["embed_used"] = counter["n_embed"]
        _atomic_write(C35.FIELD_JSON, results)
    return results


# ==========================================================================
# --fit : model_fit internals + H4 degeneracy + the null arbiter (0 calls)
# ==========================================================================
def run_fit(base_dir="."):
    """Fit the G0 double-well to the measured R3 field (model_fit internals,
    no code copied), evaluate the FROZEN H4 degeneracy criterion (before-span
    < DEGENERACY_SPAN), and embed the null-arbiter summary. The fit sign is
    NEVER reported without the arbiter (contract_g35 -- the a=+241 llama31
    artifact is the precedent): a missing arbiter file REFUSES the fit."""
    from model_fit import load_field_points, fit_model
    from field_g2 import LEVELS          # the frozen 5-level recipe order

    field_path = os.path.join(base_dir, C35.FIELD_JSON)
    if not os.path.exists(field_path):
        raise SystemExit(f"{field_path} absent -- run --field first")
    arb_path = os.path.join(base_dir, C35.ARBITER_NULL_JSON)
    if not os.path.exists(arb_path):
        raise SystemExit(
            f"REFUSING --fit: the null arbiter {arb_path} is absent and the "
            f"fit sign is NEVER reported without it (contract_g35; the "
            f"a=+241 llama31 artifact is the precedent).")

    pts, field = load_field_points(field_path)
    model = fit_model(pts)
    a = model["a"]

    before_values = {lv: field["levels"][lv]["before"]
                     for lv in LEVELS if lv in field["levels"]}
    vals = list(before_values.values())
    before_span = float(max(vals) - min(vals)) if vals else None
    degenerate = (None if before_span is None
                  else bool(before_span < C35.DEGENERACY_SPAN))

    with open(arb_path, "r", encoding="utf-8") as fh:
        arb = json.load(fh)
    arbiter = {
        "source": C35.ARBITER_NULL_JSON,
        "r": arb["observed_memory_r"],
        "abs_r": arb["observed_memory_abs_r"],
        "percentile": arb["observed_memory_percentile"],
        "pct95_abs_r": arb["pct95_abs_r"],
        "v1_pass": bool(arb["V1_path_dependence_beyond_null"]),
    }

    out = {
        "rung": C35.RUNG,
        "field_source": field_path,
        "a": a, "b": model["b"], "drive": model["drive"],
        "sse": model["sse"], "method": model["method"],
        "monostable": bool(a < 0),   # a<0 monostable; a>0 bistable/double-well
        "before_values": before_values,
        "before_span": before_span,
        "degeneracy_span_threshold": C35.DEGENERACY_SPAN,
        "degenerate": degenerate,
        "arbiter": arbiter,
        "interpretation": (
            "NOT-INTERPRETED (degenerate field; fit sign reported only "
            "alongside the arbiter)" if degenerate
            else "see supervisor ruling"),
    }
    out_path = os.path.join(base_dir, C35.FIT_JSON)
    _atomic_write(out_path, out)

    print("=" * 66)
    print(f"FIELD FIT [{C35.RUNG}] (G3.5; ruling = supervisor)")
    print("=" * 66)
    print(f"  field source: {field_path}")
    print(f"  method: {model['method']}")
    print(f"  a={a:+.4f}  b={model['b']:+.4f}  SSE={model['sse']:.5f}")
    print(f"  drives: permissive={model['drive']['permissive']:+.4f}  "
          f"caution={model['drive']['caution']:+.4f}  "
          f"neutral={model['drive']['neutral']:+.4f}")
    print(f"  monostable (a<0): {out['monostable']}")
    bv = "  ".join(f"{lv}:{v:.3f}" for lv, v in before_values.items())
    print(f"  level before values: {bv}")
    deg_word = ("DEGENERATE (level collapse -- no spatial leverage; fitted "
                "params NOT interpreted)" if degenerate else "NOT degenerate")
    print(f"  H4 degeneracy: before-span={_fmt(before_span)} < "
          f"{C35.DEGENERACY_SPAN} -> {deg_word}")
    print(f"  arbiter ({arbiter['source']}): r={arbiter['r']:+.3f} at the "
          f"{arbiter['percentile']:.1f}th pct of its null "
          f"(V1 {'PASS' if arbiter['v1_pass'] else 'FAIL'}) -- the fit sign "
          f"is reported ONLY alongside this")
    print(f"  interpretation: {out['interpretation']}")
    print("=" * 66)
    print(f"written: {out_path}")
    return out


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="G3.5 hysteresis + field (B)")
    ap.add_argument("--hyst", action="store_true",
                    help="run the 9-life hysteresis protocol on R3")
    ap.add_argument("--analyze", action="store_true",
                    help="frozen metrics + verdict + figures (0 LLM calls)")
    ap.add_argument("--field", action="store_true",
                    help="measure the R3 field (field_g2 verbatim, injected)")
    ap.add_argument("--fit", action="store_true",
                    help="double-well fit + H4 degeneracy + arbiter (0 calls)")
    ap.add_argument("--stub", action="store_true",
                    help="deterministic dry rehearsal (no daemon, no "
                         "memory_g3/runner_g25 imports)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    n_modes = sum(bool(x) for x in (args.hyst, args.analyze, args.field,
                                    args.fit))
    if n_modes == 0:
        ap.error("pick one of --hyst / --analyze / --field / --fit")
    if n_modes > 1:
        ap.error("run --hyst / --analyze / --field / --fit separately")

    if args.hyst:
        run_hyst(args.stub, args.resume)
    elif args.analyze:
        if args.stub:
            print("(--stub is ignored for --analyze: it reads checkpoints)")
        run_analyze()
    elif args.field:
        run_field(args.stub, args.resume)
    elif args.fit:
        if args.stub:
            print("(--stub is ignored for --fit: it reads the field JSON)")
        run_fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
