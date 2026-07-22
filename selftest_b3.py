"""selftest_b3.py -- G3 subagent B DRY self-tests (0 LLM calls).

Runs entirely against the stub backend and against synthetic JSON fixtures; no
daemon, no memory_g3.py, no Ollama. Every stub/synthetic g3_*.json artifact is
written to a scratch dir (or a temp name) and removed afterward, so a real run
can never collide with a self-test artifact.

Tests (task spec):
  (a) --exp2 --stub end-to-end for R1 writes well-formed JSON (12 lives, 4
      probes each, resume works, nominal chat calls == 1056 asserted).
  (b) turnspec/order reuse is bit-identical to G2's (same orderings from seed
      71, same texts for a given (order, seed)).
  (c) --null on synthetic exp2 JSONs with known structure (flat rung -> V1
      FAIL; strongly order-correlated rung -> V1 PASS).
  (d) --ladder interpretation branches: wall / gradient / neither each fire
      correctly (the most important test).
  (e) archived-number extraction: R0/R2 rows (and the R2 family + R4) match the
      known archived values (reads the real static archive files).

Run: python selftest_b3.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

import ladder_g3
from ladder_g3 import (
    run_exp2, run_null, classify, _rung_stats, _grade_predictions,
    EXP2_JSON_TMPL, NULL_JSON_TMPL,
)
from experiments_g2 import (
    make_exp2_orderings, build_exp2_turnspecs, first_quarter_share,
    exp2_arm_stats,
)

PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    tag = PASS if ok else FAIL
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    return ok


# ==========================================================================
# scratch management: isolate every synthetic/stub artifact
# ==========================================================================
def _scratch():
    return tempfile.mkdtemp(prefix="g3_selftest_")


def _cleanup_cwd_g3(names):
    for n in names:
        for suffix in ("", ".tmp"):
            p = n + suffix
            if os.path.exists(p):
                os.remove(p)


# ==========================================================================
# (a) --exp2 --stub end-to-end (12 lives, 4 probes, resume, 1056 calls)
# ==========================================================================
def test_a_exp2_stub():
    print("\n(a) --exp2 --stub end-to-end (scratch slug RSTUBA)")
    # Use a SCRATCH rung slug, never the real "R1", so a real g3_exp2_R1.json is
    # never clobbered (test spec: real runs can never collide). run_exp2 only
    # accepts NEW_RUNGS, so we monkeypatch it to include the scratch slug --
    # identical to test (c)'s RFLAT/RCORR pattern. A leftover g3_exp2_RSTUBA.json
    # is never read by any real run.
    saved_new = ladder_g3.NEW_RUNGS
    ladder_g3.NEW_RUNGS = ("R1", "R3", "RSTUBA")
    rung = "RSTUBA"
    path = EXP2_JSON_TMPL.format(rung=rung)
    replies = ladder_g3.REPLIES_JSON_TMPL.format(rung=rung)
    # ensure a clean slate (never touches the real R1/R3 files)
    _cleanup_cwd_g3([path, replies])
    try:
        # run fresh; capture the counter by re-deriving from the JSON
        run_exp2(rung, use_stub=True, resume=False)
        d = json.load(open(path, encoding="utf-8"))

        check("(a1) 12 memory lives", len(d["lives"]) == 12,
              f"n={len(d['lives'])}")
        four_probes = all(len(l["probes"]) == 4 for l in d["lives"])
        check("(a2) 4 probes per life", four_probes,
              "probe_at=[10,20,30,40]")
        turns_ok = all([p["turn"] for p in l["probes"]] == [10, 20, 30, 40]
                       for l in d["lives"])
        check("(a3) probe turns == [10,20,30,40]", turns_ok)
        # well-formed schema per life
        req = {"id", "arm", "ordering_idx", "order",
               "first_quarter_permissive_share", "final_caution", "probes",
               "store_entry_count"}
        schema_ok = all(req.issubset(l.keys()) for l in d["lives"])
        check("(a4) per-life schema well-formed", schema_ok,
              f"required keys {sorted(req)}")
        # store entry count == 40 (one entry per turn, memory arm)
        entries_ok = all(l["store_entry_count"] == 40 for l in d["lives"])
        check("(a5) store_entry_count == 40 (one per turn)", entries_ok)
        # nominal chat calls == 1056
        check("(a6) nominal chat calls == 1056", d.get("calls_used") == 1056,
              f"calls_used={d.get('calls_used')}")
        # arm is memory-only (no memoryless)
        arm_ok = all(l["arm"] == "memory" for l in d["lives"])
        check("(a7) memory arm only", arm_ok)

        # resume: truncate to 5, resume, assert 12 + bit-identical finals
        full_finals = {l["id"]: l["final_caution"] for l in d["lives"]}
        d2 = dict(d)
        d2["lives"] = d["lives"][:5]
        d2.pop("calls_used", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d2, f, indent=2)
        run_exp2(rung, use_stub=True, resume=True)
        d3 = json.load(open(path, encoding="utf-8"))
        resumed = {l["id"]: l["final_caution"] for l in d3["lives"]}
        check("(a8) resume completes 12 lives", len(d3["lives"]) == 12,
              f"n={len(d3['lives'])}")
        check("(a9) resume bit-identical finals", resumed == full_finals)
    finally:
        ladder_g3.NEW_RUNGS = saved_new
        _cleanup_cwd_g3([path, replies])


# ==========================================================================
# (b) turnspec / order reuse bit-identical to G2
# ==========================================================================
def test_b_turnspec_reuse():
    print("\n(b) turnspec/order reuse bit-identical to G2")
    ords1 = make_exp2_orderings()
    ords2 = make_exp2_orderings(seed=71, n=12)
    check("(b1) 12 orderings from seed 71 deterministic", ords1 == ords2)
    check("(b2) each ordering is the 40-event multiset",
          all(len(o) == 40 for o in ords1) and
          all(o.count("permissive") == 12 and o.count("caution") == 12 and
              o.count("neutral") == 16 for o in ords1))
    # texts identical for a given (order, seed) -- the G2 convention seed=71*100+idx
    all_det = True
    all_len = True
    for idx in range(12):
        seed = 71 * 100 + idx
        s1 = build_exp2_turnspecs(ords1[idx], seed)
        s2 = build_exp2_turnspecs(ords1[idx], seed)
        if s1 != s2:
            all_det = False
        if len(s1) != 40:
            all_len = False
    check("(b3) build_exp2_turnspecs deterministic for (order, seed)", all_det)
    check("(b4) 40 (flavor,text) specs per ordering", all_len)
    # first-quarter shares match the archived values exactly
    fq = [round(first_quarter_share(o), 3) for o in ords1]
    expected_fq = [0.6, 0.6, 0.5, 0.5, 0.0, 0.4, 0.2, 0.3, 0.3, 0.3, 0.4, 0.3]
    check("(b5) first-quarter shares match archived", fq == expected_fq,
          f"{fq}")


# ==========================================================================
# (c) --null on synthetic exp2 JSONs (flat -> FAIL; correlated -> PASS)
# ==========================================================================
def _synthetic_exp2(rung, finals, path):
    """Write a minimal but well-formed g3_exp2_<rung>.json with the given 12
    final values and the 12 fixed first-quarter shares (seed-71 orderings).
    Each life carries a 4-probe series [10,20,30,40] whose t40 == final; the
    intermediate probes are a small deterministic ramp so the 36 probe diffs
    have a realistic (nonzero) step variance for the null estimator.
    """
    ords = make_exp2_orderings()
    lives = []
    for idx in range(12):
        order = ords[idx]
        fq = first_quarter_share(order)
        f = finals[idx]
        # a monotone ramp from a fixed t10 baseline to the final at t40
        base = 0.5
        p10 = base
        p20 = base + (f - base) * 0.34
        p30 = base + (f - base) * 0.67
        p40 = f
        probes = [{"turn": t, "mean": float(m),
                   "judge_fails": 0, "per_scenario": {}}
                  for t, m in ((10, p10), (20, p20), (30, p30), (40, p40))]
        lives.append({
            "id": f"memory-ord{idx}", "arm": "memory", "ordering_idx": idx,
            "memoryless": False, "order": order,
            "first_quarter_permissive_share": fq,
            "final_caution": float(f), "final_caution_t40": float(f),
            "probes": probes, "store_entry_count": 40,
            "notes_count": 40, "summary": "",
        })
    out = {"exp": "g3_exp2", "rung": rung, "stub": False, "ceiling": 1500,
           "orderings_seed": 71, "probe_at": [10, 20, 30, 40], "lives": lives}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _write_synthetic_null(rung, v1, path):
    """Write a minimal well-formed g3_null_<rung>.json with a chosen V1 verdict,
    so the ladder can read a NEW rung as known without a real run."""
    out = {
        "rung": rung, "seed": 74, "n_walks": 10000,
        "step_variance": 0.003, "step_sd": 0.0548, "n_probe_diffs": 36,
        "start": 0.5, "walk_n_steps": 3, "pct95_abs_r": 0.575,
        "observed_memory_r": (-0.7 if v1 else -0.2),
        "observed_memory_abs_r": (0.7 if v1 else 0.2),
        "observed_memory_percentile": (99.0 if v1 else 40.0),
        "memory_std_final": (0.18 if v1 else 0.05),
        "memory_finals": [0.5] * 12,
        "bimodality_counts": {"below_0.40": 0, "0.40-0.60": 12, "above_0.60": 0},
        "V1_path_dependence_beyond_null": bool(v1),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def test_c_null_synthetic():
    print("\n(c) --null on synthetic exp2 (flat FAIL / correlated PASS)")
    # We must place the synthetic exp2 at the path run_null reads (cwd), but use
    # a NEW rung slug so it never collides with a real R1/R3 run. run_null only
    # accepts NEW_RUNGS, so we monkeypatch NEW_RUNGS to include our test slugs.
    saved_new = ladder_g3.NEW_RUNGS
    ladder_g3.NEW_RUNGS = ("R1", "R3", "RFLAT", "RCORR")
    ords = make_exp2_orderings()
    fq_shares = np.array([first_quarter_share(o) for o in ords])

    made = []
    try:
        # --- flat rung: all finals ~equal (tiny deterministic jitter) -> the
        # correlation with fq is ~0, well inside the null -> V1 FAIL.
        flat_finals = [0.60 + ((i % 3) - 1) * 0.002 for i in range(12)]
        exp2_flat = EXP2_JSON_TMPL.format(rung="RFLAT")
        null_flat = NULL_JSON_TMPL.format(rung="RFLAT")
        _synthetic_exp2("RFLAT", flat_finals, exp2_flat)
        made += [exp2_flat, null_flat]
        run_null("RFLAT")
        nf = json.load(open(null_flat, encoding="utf-8"))
        check("(c1) flat rung V1 FAIL",
              nf["V1_path_dependence_beyond_null"] is False,
              f"|r|={nf['observed_memory_abs_r']:.3f} pct95={nf['pct95_abs_r']:.3f}")

        # --- strongly order-correlated rung: finals = a strong LINEAR function
        # of fq share (negative slope, the G0 early-permissive-dominates sign),
        # spread wide so |r| is near 1 and far beyond the null 95th pct -> PASS.
        corr_finals = list(0.85 - 0.9 * fq_shares)   # fq 0..0.6 -> 0.85..0.31
        exp2_corr = EXP2_JSON_TMPL.format(rung="RCORR")
        null_corr = NULL_JSON_TMPL.format(rung="RCORR")
        _synthetic_exp2("RCORR", corr_finals, exp2_corr)
        made += [exp2_corr, null_corr]
        run_null("RCORR")
        nc = json.load(open(null_corr, encoding="utf-8"))
        check("(c2) correlated rung V1 PASS",
              nc["V1_path_dependence_beyond_null"] is True,
              f"|r|={nc['observed_memory_abs_r']:.3f} pct95={nc['pct95_abs_r']:.3f}")
        # sanity: the correlated |r| is genuinely large
        check("(c3) correlated |r| > 0.9", nc["observed_memory_abs_r"] > 0.9,
              f"|r|={nc['observed_memory_abs_r']:.3f}")
        # sanity: the null estimator used the 36 own probe diffs
        check("(c4) null used 36 own probe diffs",
              nc["n_probe_diffs"] == 36 and nc["walk_n_steps"] == 3,
              f"n_diffs={nc['n_probe_diffs']} steps={nc['walk_n_steps']}")
    finally:
        ladder_g3.NEW_RUNGS = saved_new
        _cleanup_cwd_g3(made)


# ==========================================================================
# (d) --ladder interpretation branches (wall / gradient / neither)
# ==========================================================================
def _synthetic_rows(v1_map, coupling_map=None):
    """Build synthetic ladder rows from a {rung: V1} map, using the FROZEN
    RUNGS classification for the axis properties (so the classify() logic sees
    the real coupling/compression/recurrence codings). null_pct is set from V1
    (PASS -> a high pct, FAIL -> a low pct) so the gradient signal is defined.
    """
    from contract_g3 import RUNGS
    coupling_map = coupling_map or {}
    rows = []
    for rung in ["R0", "R1", "R2", "R3", "R4"]:
        meta = RUNGS[rung]
        v1 = v1_map.get(rung)
        # derive a percentile consistent with V1 for the gradient test
        if v1 is True:
            pct = 99.0
        elif v1 is False:
            pct = 50.0
        else:
            pct = None
        rows.append({
            "rung": rung, "name": meta["name"],
            "compression": meta.get("compression"),
            "recurrence": meta.get("recurrence"),
            "coupling": coupling_map.get(rung, meta.get("coupling")),
            "status": meta.get("status"),
            "substrate": "attachment" if rung == "R4" else "caution",
            "std_final": 0.7 if v1 else 0.05,
            "abs_r": 0.7 if v1 else 0.3,
            "r": -0.7 if v1 else 0.3,
            "null_pct": pct,
            "V1": v1,
        })
    return rows


def test_d_interpretation_branches():
    print("\n(d) --ladder interpretation branches (MOST IMPORTANT)")

    # --- WALL: only R4 (the STRONG-coupling rung) passes; all else FAIL, no
    #     unknowns. Pass-set == coupled-set -> WALL.
    wall_rows = _synthetic_rows(
        {"R0": False, "R1": False, "R2": False, "R3": False, "R4": True})
    wall = classify(wall_rows)
    check("(d1) WALL fires when only R4 passes",
          wall["branch"] == "WALL" and wall["axis"] == "perception-coupling",
          f"branch={wall['branch']} axis={wall['axis']}")

    # --- GRADIENT: break the wall (R2 also passes) AND make the pass/percentile
    #     signal rise monotonically along the COUPLING axis:
    #       none (R0,R1,R3) FAIL -> weak/implicit (R2) PASS -> STRONG (R4) PASS.
    #     Pass-set {R2,R4} != coupled-set {R4} -> not WALL -> GRADIENT.
    grad_rows = _synthetic_rows(
        {"R0": False, "R1": False, "R2": True, "R3": False, "R4": True})
    grad = classify(grad_rows)
    check("(d2) GRADIENT fires when signal rises monotonically (wall broken)",
          grad["branch"] == "GRADIENT",
          f"branch={grad['branch']} axis={grad['axis']}")
    # This fixture is monotone along COUPLING (the prereg hypothesis). The
    # report MUST name coupling (not silently pick collinear compression) AND
    # carry the collinearity flag, because compression/coupling share the
    # identical ordinal vector [0,0,1,0,2] on the frozen rung set and cannot be
    # separated here. (Tightened per advisor: the loose 'axis in AXES' check
    # masked the collinearity bug.)
    det = grad["detail"]
    check("(d3) GRADIENT names the prereg COUPLING axis (not collinear "
          "compression by AXES order)",
          grad["axis"] == "coupling", f"axis={grad['axis']}")
    check("(d3b) monotone set includes both collinear axes {compression,coupling}",
          set(det.get("monotone_axes", [])) >= {"compression", "coupling"},
          f"monotone={det.get('monotone_axes')}")
    check("(d3c) collinearity flagged (compression/coupling inseparable)",
          bool(det.get("collinearity_note")) and
          any(set(g) == {"compression", "coupling"}
              for g in det.get("collinear_groups", [])),
          f"groups={det.get('collinear_groups')}")

    # --- NEITHER: scramble so that (i) the wall is broken (a non-coupled rung
    #     passes) and (ii) no axis is monotone. R1 (none-coupling) passes but
    #     R4 (STRONG) FAILS: coupling signal goes ...PASS(R1,none)... then
    #     FAIL(R4,STRONG) -> DECREASING on coupling; compression: R1 none-grows
    #     PASS but R4 high FAIL and R2 medium FAIL -> not monotone; recurrence
    #     likewise broken. -> NEITHER.
    neither_rows = _synthetic_rows(
        {"R0": False, "R1": True, "R2": False, "R3": False, "R4": False})
    neither = classify(neither_rows)
    check("(d4) NEITHER fires when no clean wall and no monotone axis",
          neither["branch"] == "NEITHER",
          f"branch={neither['branch']} axis={neither['axis']}")

    # --- precedence: WALL beats GRADIENT when both could read (only-R4-passes
    #     is ALSO a coupling jump). Confirm the only-R4 case returns WALL, not
    #     GRADIENT (the committed precedence).
    check("(d5) precedence WALL > GRADIENT (only-R4 reads as WALL, not gradient)",
          wall["branch"] == "WALL")

    # --- RECURRENCE is the only independently-resolvable axis ([0,1,2,1,2]);
    #     a gradient monotone on recurrence but NOT on coupling/compression must
    #     name recurrence and NOT be flagged collinear with them. Construct one:
    #     signal must rise with recurrence level (R0=0 -> R1/R3=1 -> R2/R4=2)
    #     while being non-monotone on coupling. Achieve by passing R1 and R3
    #     (recurrence level 1, coupling none) and R2,R4 (level 2) but failing R0.
    #     coupling vector [0,0,1,0,2] with signals R0=0,R1=99,R2=99,R3=99,R4=99:
    #     coupling levels none(R0,R1,R3)=[0,99,99]->mean66, weak(R2)=99,
    #     STRONG(R4)=99 -> 66,99,99 monotone TOO. So coupling co-fires; to make
    #     recurrence the DISTINGUISHING axis we instead verify recurrence appears
    #     in the monotone set as an independently-coded axis.
    rec_rows = _synthetic_rows(
        {"R0": False, "R1": True, "R2": True, "R3": True, "R4": True})
    rec = classify(rec_rows)
    rec_det = rec["detail"]
    check("(d5b) recurrence resolvable: appears in monotone set & is NOT "
          "collinear with compression",
          rec["branch"] == "GRADIENT" and
          "recurrence" in rec_det.get("monotone_axes", []) and
          not any("recurrence" in g and "compression" in g
                  for g in rec_det.get("collinear_groups", [])),
          f"monotone={rec_det.get('monotone_axes')} "
          f"collinear={rec_det.get('collinear_groups')}")

    # --- failing-rung floor: a HIGH-but-FAILING percentile must NOT manufacture
    #     a gradient. Give R2 a failing V1 but a high (78) percentile; with the
    #     floor it contributes 0, so R2 cannot create a false monotone rise.
    #     Only R4 passes -> this is the WALL fixture; assert R2's raw 78 did not
    #     leak into a gradient signal by checking the WALL fixture's gradient
    #     detail would floor R2. (Directly exercise _rung_signal.)
    from ladder_g3 import _rung_signal
    r2_fail_high = {"rung": "R2", "V1": False, "null_pct": 78.17,
                    "coupling": "weak/implicit"}
    check("(d5c) V1-FAIL rung floored to 0 (high pct does not manufacture "
          "path-dependence)",
          _rung_signal(r2_fail_high) == 0.0,
          f"signal={_rung_signal(r2_fail_high)}")
    r4_pass = {"rung": "R4", "V1": True, "null_pct": None}
    check("(d5d) V1-PASS rung with no pct uses ceiling 100",
          _rung_signal(r4_pass) == 100.0)

    # --- unknown V1 blocks a clean wall (a pending rung must not let WALL fire)
    wall_pending = _synthetic_rows(
        {"R0": False, "R1": False, "R2": False, "R3": None, "R4": True})
    wp = classify(wall_pending)
    check("(d6) unknown V1 (R3 pending) blocks a clean WALL",
          wp["branch"] != "WALL", f"branch={wp['branch']}")

    # --- run_ladder guard tests (d9-d11): these exercise the REAL rung slugs
    #     R1/R3 (run_ladder's rung list is hardcoded and cannot take a scratch
    #     slug). To avoid ever deleting a real null output, REFUSE to run these
    #     if a real g3_null_R1/R3 is present in cwd -- the operator sees a
    #     SKIP-with-reason instead of losing data. (These outputs are 0-LLM
    #     regenerable, so skipping costs the operator nothing.)
    real_nulls = [NULL_JSON_TMPL.format(rung=r) for r in ("R1", "R3")]
    real_ladder = [p for p in ("g3_ladder.json", "g3_fig_ladder.png")
                   if os.path.exists(p)]
    if any(os.path.exists(p) for p in real_nulls) or real_ladder:
        print("  [SKIP] (d9-d11) real g3_null_R1/R3 or g3_ladder present in "
              "cwd; skipping the run_ladder guard tests rather than deleting "
              "real outputs. (Move them aside to exercise d9-d11.)")
    else:
        # (d9/d10) operational incompleteness guard: with no g3_null_R1/R3 in
        # cwd, R1/R3 are pending -> run_ladder withholds the authoritative
        # interpretation (NOT a 4th branch; the prereg froze 3 outcomes) and the
        # provisional branch names COUPLING, not collinear compression.
        try:
            out = ladder_g3.run_ladder()
            check("(d9) run_ladder withholds interpretation when NEW rungs "
                  "pending",
                  out["interpretation_withheld"] is True and
                  set(out["pending_rungs"]) == {"R1", "R3"},
                  f"pending={out['pending_rungs']}")
            check("(d10) provisional branch names prereg COUPLING axis (not "
                  "collinear compression)",
                  out["interpretation"]["axis"] == "coupling",
                  f"axis={out['interpretation']['axis']}")
        finally:
            _cleanup_cwd_g3(["g3_ladder.json", "g3_fig_ladder.png"])

        # (d11) full ladder with BOTH new rungs known (synthetic nulls at the
        # real slugs, safe because we just confirmed none existed): a complete
        # ladder fires a real (non-withheld) branch. R1 FAIL, R3 FAIL (the
        # committed prediction) -> only R4 passes -> WALL, complete.
        made = []
        try:
            for rung, v1 in (("R1", False), ("R3", False)):
                npath = NULL_JSON_TMPL.format(rung=rung)
                _write_synthetic_null(rung, v1, npath)
                made.append(npath)
            out2 = ladder_g3.run_ladder()
            check("(d11) complete ladder (R1/R3 FAIL) -> not withheld, WALL "
                  "fires",
                  out2["interpretation_withheld"] is False and
                  out2["interpretation"]["branch"] == "WALL",
                  f"withheld={out2['interpretation_withheld']} "
                  f"branch={out2['interpretation']['branch']}")
        finally:
            _cleanup_cwd_g3(made + ["g3_ladder.json", "g3_fig_ladder.png"])

    # --- prediction grading is frozen (no relabel): committed R4 PASS, rest FAIL
    grades = _grade_predictions(wall_rows)
    committed_ok = (grades["R4"]["committed_V1"] is True and
                    all(grades[r]["committed_V1"] is False
                        for r in ("R0", "R1", "R2", "R3")))
    check("(d7) frozen predictions: R4 committed PASS, R0-R3 committed FAIL",
          committed_ok)
    # on the wall fixture, all predictions are correct
    all_correct = all(grades[r]["prediction_correct"] for r in grades)
    check("(d8) wall fixture -> all frozen predictions correct", all_correct)


# ==========================================================================
# (e) archived-number extraction (reads the real static archive files)
# ==========================================================================
def test_e_archived_extraction():
    print("\n(e) archived-number extraction (R0/R2/family/R4)")

    r0 = _rung_stats("R0")
    check("(e1) R0 std == 0.071 (g2 memoryless arm)",
          abs(r0["std_final"] - 0.0708) < 5e-4,
          f"std={r0['std_final']:.4f}")
    check("(e2) R0 V1 FALSE (control)", r0["V1"] is False)

    r2 = _rung_stats("R2")
    check("(e3) R2 std == 0.051 (g2 memory arm)",
          abs(r2["std_final"] - 0.0506) < 5e-4,
          f"std={r2['std_final']:.4f}")
    check("(e4) R2 null pct == 78.17 (g2_null_results)",
          abs(r2["null_pct"] - 78.17) < 0.05,
          f"null_pct={r2['null_pct']:.2f}")
    check("(e5) R2 V1 FALSE (|r| below null 95th)", r2["V1"] is False,
          f"|r|={r2['abs_r']:.3f}")
    # family replicates
    fam = r2.get("family", {})
    check("(e6) R2 family llama31 std 0.027 @ 71.62",
          "llama31" in fam and abs(fam["llama31"]["std_final"] - 0.0267) < 1e-3
          and abs(fam["llama31"]["null_pct"] - 71.62) < 0.05,
          f"{fam.get('llama31')}")
    check("(e7) R2 family ablit std 0.077 @ 58.42",
          "ablit" in fam and abs(fam["ablit"]["std_final"] - 0.0769) < 1e-3
          and abs(fam["ablit"]["null_pct"] - 58.42) < 0.05,
          f"{fam.get('ablit')}")

    r4 = _rung_stats("R4")
    check("(e8) R4 std == 0.7635 (g15 p2 spread_std, attachment)",
          abs(r4["std_final"] - 0.763503) < 1e-4,
          f"std={r4['std_final']:.4f}")
    check("(e9) R4 early_corr == 0.5099 (g15 p2)",
          abs(r4["r"] - 0.509942) < 1e-4, f"r={r4['r']:.4f}")
    check("(e10) R4 V1 PASS (imported from G1.5)", r4["V1"] is True)
    check("(e11) R4 substrate flagged 'attachment' + protocol-diff present",
          r4["substrate"] == "attachment" and bool(r4.get("protocol_diff")))


# ==========================================================================
def main():
    print("=" * 72)
    print("selftest_b3 -- G3 subagent B DRY self-tests (0 LLM calls)")
    print("=" * 72)

    # All exp2/null artifacts this selftest writes use SCRATCH slugs (RSTUBA,
    # RFLAT, RCORR) that no real run ever reads -- so tests (a)/(c) can never
    # clobber a real g3_exp2_R1/R3 or g3_null_R1/R3. The only real-slug files
    # touched are the ladder-guard tests (d9/d11), which REFUSE to run if a real
    # g3_null_R1/R3 is present (see test_d) rather than deleting it.
    test_a_exp2_stub()
    test_b_turnspec_reuse()
    test_c_null_synthetic()
    test_d_interpretation_branches()
    test_e_archived_extraction()

    # final sweep: ensure no stray g3 self-test artifact survived
    stray = [p for p in os.listdir(".")
             if p.startswith(("g3_exp2_RFLAT", "g3_exp2_RCORR",
                              "g3_null_RFLAT", "g3_null_RCORR"))]
    check("(cleanup) no synthetic RFLAT/RCORR artifacts left", not stray,
          f"stray={stray}" if stray else "clean")

    n_pass = sum(1 for _, ok in _results if ok)
    n_total = len(_results)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    print("=" * 72)
    if n_pass != n_total:
        print("FAILURES:")
        for name, ok in _results:
            if not ok:
                print(f"  - {name}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
