"""field_g2.py -- G2 subagent C: the field method + fidelity check (bridge).

CLI:
    python field_g2.py --measure  [--resume] [--stub]
    python field_g2.py --fidelity [--resume] [--stub]

The FIELD METHOD applied to the natural agent (G2.md §5): measure the caution
transition Delta = caution_after - caution_before as a function of (current
caution level, pressure type). Fit later by model_fit.py. The FIDELITY CHECK
(prereg PRED-5, gates the shortcut) then compares a few field-simulated
trajectories against real full-agent lives; if they diverge the drift is not
scalar-summarizable and the supervisor falls back to real trajectories.

Zero LLM calls originate here. Drives A's agent loop + B's battery through the
same backend seam as experiments_g2 (real or --stub). Imitates the G1.5 field /
fidelity PATTERNS (experiments_field.py, fidelity_check.py) but on a NEW axis --
nothing imported from those files.

DECIDE-AND-DOCUMENT (prereg gives latitude; frozen here, not escalated):
  * The 5-level PREFIX RECIPE (below) -- fixed sequences of pressure turns
    applied from a fresh store to prepare 5 distinct caution levels.
  * Rep variation: the 5 reps per (level, type) differ by the pressure-TEXT
    variant used for the single applied turn (cycled through the contract's 5
    texts per flavor by rep index). Same before-snapshot, different text -> a
    spread of Delta at fixed level. Documented per the task brief.
  * Field-sim baseline: caution state c starts at the L3 (neutral-prefix)
    measured BEFORE value -- the closest thing to an unpressured baseline in the
    grid. Documented; see field_sim().
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contract_g2 import MODEL, PRESSURE

# reuse the backend seam + counting from experiments_g2 (pure imports, no LLM)
from experiments_g2 import (
    CountingClient, make_backend, _atomic_write, _load_results,
    _guard_resume_mode,
)

plt.style.use("dark_background")

FIELD_JSON = "g2_field.json"
FIDELITY_JSON = "g2_fidelity.json"
FIDELITY_FIG = "g2_fig_fidelity.png"

MEASURE_CEILING = 1200

# --------------------------------------------------------------------------
# 5-level prefix recipe (FROZEN here). Each is a list of pressure flavors run
# from a fresh store to prepare a distinct caution level. Texts cycle in
# contract order within a prefix. Chosen to span cavalier -> cautious:
#   L1: 8 permissive        (deep cavalier)
#   L2: 4 permissive        (mildly cavalier)
#   L3: 4 neutral           (near baseline)
#   L4: 4 caution           (mildly cautious)
#   L5: 8 caution           (deep cautious)
# --------------------------------------------------------------------------
PREFIX_RECIPE = {
    "L1": ["permissive"] * 8,
    "L2": ["permissive"] * 4,
    "L3": ["neutral"] * 4,
    "L4": ["caution"] * 4,
    "L5": ["caution"] * 8,
}
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
PRESSURE_TYPES = ["permissive", "caution", "neutral"]
REPS_PER_CELL = 5

# fidelity (prereg EXACT)
FIDELITY_SHARES = [0.1, 0.3, 0.5, 0.7, 0.9]
FIDELITY_N_PRESSURE = 24       # pressure turns per life
FIDELITY_N_NEUTRAL = 16
FIDELITY_TURNS = FIDELITY_N_PRESSURE + FIDELITY_N_NEUTRAL   # 40
FIDELITY_PROBE_AT = [10, 20, 30, 40]
FIDELITY_SEQ_SEED = 73
FIELD_SIM_NOISE_SEED = 72

# PRED-5 pass-lines (prereg -- FROZEN)
PRED_5A_MEAN_TOL = 0.15
PRED_5B_PAIR_TOL = 0.25
PRED_5B_MIN_PAIRS = 4          # >= 4/5


# ==========================================================================
# helpers: build a prepared store, run one pressure turn, probe
# ==========================================================================
def _cycle_text(flavor, idx):
    texts = PRESSURE[flavor]
    return texts[idx % len(texts)]


def _build_prefix_store(backend, flavors):
    """Run a prefix of pressure flavors from a FRESH store with the memory agent
    (memoryless=False). Returns the store. Texts cycle in contract order."""
    store = backend["make_store"]()
    counts = {"permissive": 0, "caution": 0, "neutral": 0}
    for turn_idx, fl in enumerate(flavors):
        text = _cycle_text(fl, counts[fl])
        counts[fl] += 1
        backend["agent_turn"](text, store, turn_idx, client=backend["client"],
                              model=MODEL, memoryless=False, flavor=fl)
    return store


def _battery_mean(backend, store):
    b = backend["run_battery"](store, client=backend["client"], model=MODEL)
    return float(b.mean)


# ==========================================================================
# --measure : the transition field
# ==========================================================================
def run_measure(use_stub, resume):
    counter = {"n_calls": 0}
    backend = make_backend(use_stub, counter)
    results = _load_results(FIELD_JSON) if resume else None
    _guard_resume_mode(results, use_stub, FIELD_JSON)
    if results is None:
        results = {"mode": "field", "stub": bool(use_stub),
                   "ceiling": MEASURE_CEILING, "prefix_recipe": PREFIX_RECIPE,
                   "reps_per_cell": REPS_PER_CELL, "levels": {}, "records": []}
    else:
        print(f"[resume] {FIELD_JSON}: levels done = "
              f"{list(results['levels'].keys())}")

    t0 = time.time()
    for level in LEVELS:
        if level in results["levels"]:
            continue
        # build the prepared store ONCE, probe BEFORE once, snapshot
        store = _build_prefix_store(backend, PREFIX_RECIPE[level])
        _check_measure_ceiling(counter)
        before = _battery_mean(backend, store)
        _check_measure_ceiling(counter)
        snap = store.snapshot()

        level_entry = {"before": before, "cells": {}}
        prefix_len = len(PREFIX_RECIPE[level])
        for ptype in PRESSURE_TYPES:
            deltas = []
            for rep in range(REPS_PER_CELL):
                store.restore(snap)                     # pristine prepared state
                text = _cycle_text(ptype, rep)          # rep -> text variant
                # apply ONE pressure turn at position = prefix_len (its "age")
                backend["agent_turn"](text, store, prefix_len,
                                      client=backend["client"], model=MODEL,
                                      memoryless=False, flavor=ptype)
                _check_measure_ceiling(counter)
                after = _battery_mean(backend, store)
                _check_measure_ceiling(counter)
                d = after - before
                deltas.append(d)
                results["records"].append(
                    {"level": level, "before": before, "type": ptype,
                     "rep": rep, "after": after, "delta": d})
            deltas = np.array(deltas, float)
            level_entry["cells"][ptype] = {
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std()),
                "n": int(len(deltas)),
                "deltas": deltas.tolist(),
            }
        results["levels"][level] = level_entry
        results["calls_used"] = counter["n_calls"]
        _atomic_write(FIELD_JSON, results)
        el = time.time() - t0
        cell_summary = "  ".join(
            f"{pt[:4]}:{level_entry['cells'][pt]['mean_delta']:+.3f}"
            for pt in PRESSURE_TYPES)
        print(f"  {level} before={before:.3f}  {cell_summary}  | "
              f"calls={counter['n_calls']} elapsed={el:.0f}s")

    _atomic_write(FIELD_JSON, results)
    print(f"measure done: {len(results['levels'])}/{len(LEVELS)} levels, "
          f"{counter['n_calls']} calls (ceiling {MEASURE_CEILING})")
    _print_field_table(results)
    return results


def _check_measure_ceiling(counter):
    if counter["n_calls"] > MEASURE_CEILING:
        raise RuntimeError(
            f"MEASURE CALL CEILING EXCEEDED: {counter['n_calls']} > "
            f"{MEASURE_CEILING}. Aborting before the daemon is over-used.")


def _print_field_table(results):
    print("\n" + "=" * 62)
    print("FIELD Delta table (mean Delta caution per level x pressure type)")
    print("=" * 62)
    print(f"  {'level':6s} {'before':>7s}  " +
          "  ".join(f"{pt:>10s}" for pt in PRESSURE_TYPES))
    for level in LEVELS:
        if level not in results["levels"]:
            continue
        e = results["levels"][level]
        row = f"  {level:6s} {e['before']:>7.3f}  "
        row += "  ".join(
            f"{e['cells'][pt]['mean_delta']:>+10.3f}" for pt in PRESSURE_TYPES)
        print(row)
    print("=" * 62)


# ==========================================================================
# field-sim : replay a flavor sequence over the measured Delta field
# ==========================================================================
def load_field(path=FIELD_JSON):
    """Load the measured field. Returns (before_levels sorted ascending,
    {type: mean_delta[level]}, {type: std_delta[level]}). before_levels is the
    x-axis for np.interp (the measured BEFORE caution per level)."""
    results = _load_results(path)
    if results is None:
        raise RuntimeError(f"{path} absent -- run --measure first.")
    levels = [lv for lv in LEVELS if lv in results["levels"]]
    befores = np.array([results["levels"][lv]["before"] for lv in levels], float)
    order = np.argsort(befores)             # np.interp needs ascending x
    befores = befores[order]
    means = {pt: np.array(
        [results["levels"][levels[i]]["cells"][pt]["mean_delta"] for i in order],
        float) for pt in PRESSURE_TYPES}
    stds = {pt: np.array(
        [results["levels"][levels[i]]["cells"][pt]["std_delta"] for i in order],
        float) for pt in PRESSURE_TYPES}
    return befores, means, stds, results


def field_sim(sequence, befores, means, stds, *, start, noise_seed):
    """Simulate one caution trajectory by replaying `sequence` (list of flavors)
    over the interpolated Delta field. c starts at `start`; each turn
    c += interp_Delta(c, flavor) + N(0, interp_std). Linear interpolation via
    np.interp over the measured BEFORE-caution values. Returns the final c."""
    rng = np.random.default_rng(noise_seed)
    c = float(start)
    for fl in sequence:
        d = float(np.interp(c, befores, means[fl]))
        sd = float(np.interp(c, befores, stds[fl]))
        if sd > 0:
            d += float(rng.normal(0.0, sd))
        c = float(np.clip(c + d, 0.0, 1.0))
    return c


# ==========================================================================
# --fidelity : 5 real lives vs 5 field-sims on the SAME sequences
# ==========================================================================
def _fidelity_sequences():
    """The 5 frozen fidelity flavor-sequences (prereg): permissive share in
    {0.1,0.3,0.5,0.7,0.9} of the 24 pressure turns + 16 neutral, shuffled by
    default_rng(73). Returns list of (share, sequence)."""
    rng = np.random.default_rng(FIDELITY_SEQ_SEED)
    seqs = []
    for share in FIDELITY_SHARES:
        n_perm = int(round(share * FIDELITY_N_PRESSURE))
        n_caut = FIDELITY_N_PRESSURE - n_perm
        multiset = (["permissive"] * n_perm + ["caution"] * n_caut +
                    ["neutral"] * FIDELITY_N_NEUTRAL)
        seq = list(rng.permutation(multiset))
        seqs.append((share, seq))
    return seqs


def _attach_fidelity_texts(sequence, seed):
    """Attach concrete texts: neutral drawn by `seed`, permissive/caution cycle."""
    rng = np.random.default_rng(seed)
    n_neutral_texts = len(PRESSURE["neutral"])
    specs = []
    perm_c = caut_c = 0
    for fl in sequence:
        if fl == "neutral":
            text = PRESSURE["neutral"][int(rng.integers(n_neutral_texts))]
        elif fl == "permissive":
            text = _cycle_text("permissive", perm_c); perm_c += 1
        else:
            text = _cycle_text("caution", caut_c); caut_c += 1
        specs.append((fl, text))
    return specs


def run_fidelity(use_stub, resume):
    counter = {"n_calls": 0}
    backend = make_backend(use_stub, counter)
    results = _load_results(FIDELITY_JSON) if resume else None
    _guard_resume_mode(results, use_stub, FIDELITY_JSON)
    if results is None:
        results = {"mode": "fidelity", "stub": bool(use_stub),
                   "shares": FIDELITY_SHARES, "lives": []}
    else:
        print(f"[resume] {FIDELITY_JSON}: {len(results['lives'])} lives done")
    done = {life["id"] for life in results["lives"]}

    sequences = _fidelity_sequences()
    t0 = time.time()
    for i, (share, seq) in enumerate(sequences):
        life_id = f"life{i}-share{share}"
        if life_id in done:
            continue
        seed = FIDELITY_SEQ_SEED * 100 + i
        specs = _attach_fidelity_texts(seq, seed)
        store = backend["make_store"]()
        probe_series = []
        for turn_idx, (flavor, text) in enumerate(specs):
            backend["agent_turn"](text, store, turn_idx, client=backend["client"],
                                  model=MODEL, memoryless=False, flavor=flavor)
            if (turn_idx + 1) in FIDELITY_PROBE_AT:
                probe_series.append(
                    {"turn": turn_idx + 1, "mean": _battery_mean(backend, store)})
        real_final = probe_series[-1]["mean"]
        results["lives"].append({
            "id": life_id, "share": share, "sequence": seq,
            "real_probes": probe_series, "real_final": real_final})
        results["calls_used"] = counter["n_calls"]
        _atomic_write(FIDELITY_JSON, results)
        el = time.time() - t0
        print(f"  [{i+1}/5] {life_id}: real_final={real_final:.3f} | "
              f"calls={counter['n_calls']} elapsed={el:.0f}s")

    # ---- field-sim replays of the SAME sequences (no LLM) ----
    try:
        befores, means, stds, _ = load_field()
    except RuntimeError as e:
        print(f"[field-sim SKIPPED] {e}")
        _atomic_write(FIDELITY_JSON, results)
        return results

    # baseline start = L3 (neutral-prefix) measured BEFORE value (documented)
    start = None
    field_results = _load_results(FIELD_JSON)
    if field_results and "L3" in field_results["levels"]:
        start = field_results["levels"]["L3"]["before"]
    if start is None:
        start = float(befores[len(befores) // 2])
    results["field_sim_start"] = start

    for life in results["lives"]:
        if "field_final" in life:
            continue
        life["field_final"] = field_sim(
            life["sequence"], befores, means, stds,
            start=start, noise_seed=FIELD_SIM_NOISE_SEED)
    _atomic_write(FIDELITY_JSON, results)

    evaluate_fidelity(results)
    return results


def evaluate_fidelity(results):
    """PRED-5a/b vs the frozen lines + figure. Verdict is the supervisor's."""
    lives = [l for l in results["lives"] if "field_final" in l]
    if not lives:
        print("[fidelity] no field-sims yet (field not measured?)")
        return
    real = np.array([l["real_final"] for l in lives], float)
    field = np.array([l["field_final"] for l in lives], float)
    shares = [l["share"] for l in lives]
    mean_diff = float(abs(real.mean() - field.mean()))
    pair_diffs = np.abs(real - field)
    pairs_ok = int((pair_diffs < PRED_5B_PAIR_TOL).sum())
    p5a = mean_diff < PRED_5A_MEAN_TOL
    p5b = pairs_ok >= PRED_5B_MIN_PAIRS

    print("\n" + "=" * 62)
    print("FIELD FIDELITY (PRED-5; informational; verdict = supervisor)")
    print("=" * 62)
    print(f"  mean final: real={real.mean():.3f} field={field.mean():.3f} "
          f"|diff|={mean_diff:.3f}")
    print(f"  PRED-G2-5a |mean diff| < {PRED_5A_MEAN_TOL}: {mean_diff:.3f} -> "
          f"{'PASS' if p5a else 'FAIL'}")
    print(f"  per-pair |real-field|: {np.round(pair_diffs, 3).tolist()}")
    print(f"  PRED-G2-5b |diff| < {PRED_5B_PAIR_TOL} for >= "
          f"{PRED_5B_MIN_PAIRS}/5: {pairs_ok}/5 -> {'PASS' if p5b else 'FAIL'}")
    fidelity_ok = p5a and p5b
    print(f"  -> field shortcut {'VALIDATED' if fidelity_ok else 'FAILS -- drift is higher-dimensional; use real trajectories'}")
    print("=" * 62)

    results["pred5"] = {"mean_diff": mean_diff, "pairs_ok": pairs_ok,
                        "p5a": p5a, "p5b": p5b, "fidelity_ok": fidelity_ok,
                        "pair_diffs": pair_diffs.tolist()}
    _atomic_write(FIDELITY_JSON, results)

    # figure: y=real, x=field, diagonal (task brief)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], color="#666", lw=0.8, ls="--")
    ax.scatter(field, real, s=70, c="#00e5ff", edgecolors="none")
    for i in range(len(lives)):
        ax.annotate(f"{shares[i]}", (field[i], real[i]), fontsize=8,
                    color="#aaa", xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("final caution -- field-sim")
    ax.set_ylabel("final caution -- real agent")
    verdict = "VALIDATED" if fidelity_ok else "FAILS"
    ax.set_title(f"G2 field fidelity: {verdict}\n"
                 f"mean diff {mean_diff:.2f}, pairs {pairs_ok}/5", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIDELITY_FIG, dpi=120)
    plt.close(fig)
    print(f"written: {FIDELITY_JSON}, {FIDELITY_FIG}")


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="G2 field method + fidelity (C)")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args(argv)
    if not (args.measure or args.fidelity):
        ap.error("pick --measure or --fidelity")
    if args.measure:
        run_measure(args.stub, args.resume)
    if args.fidelity:
        run_fidelity(args.stub, args.resume)


if __name__ == "__main__":
    main()
