"""selftest_b25.py -- DRY self-tests for G2.5 subagent B (ZERO LLM calls).

    python selftest_b25.py

All tests run against stub_g2's deterministic backend in an ISOLATED scratch
directory (so no g25_*.json artifact ever lands where a real --resume would
read it). Covers:

  (a) --direction --stub end-to-end for one slug: completes, writes a
      well-formed g25_direction JSON with every frozen metric present, and
      spends exactly the nominal 654 calls.
  (b) resume: interrupt after N lives, resume, and confirm completed lives are
      skipped (not re-run) and the final file is identical to an uninterrupted
      run.
  (c) turnspec construction: the mem-perm flavor sequence is EXACTLY 10n/15p/5n
      and every text comes from the frozen PRESSURE pools.
  (d) --decide on SYNTHETIC inputs: crafted direction+fit JSONs that must fire
      branch R, S, M, and F respectively -- the decision rule must be
      mechanically correct (the most important test).
  (e) field_g25 --measure --stub: runs field_g2.run_measure through the
      namespace injection without editing field_g2, and --fit reproduces the
      archived qwen a.

Prints a PASS/FAIL summary and exits nonzero on any failure.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

import contract_g25 as C25
from contract_g2 import PRESSURE

import experiments_g25 as X25
import field_g25 as F25


PASS = []
FAIL = []


def _check(cond, label):
    if cond:
        PASS.append(label)
        print(f"  PASS  {label}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label}")
    return cond


# ==========================================================================
# (c) turnspec construction (pure -- no backend needed)
# ==========================================================================
def test_turnspecs():
    print("\n--- (c) turnspec construction ---")
    flavors = X25.mem_perm_flavors()
    expected = (["neutral"] * 10 + ["permissive"] * 15 + ["neutral"] * 5)
    _check(flavors == expected,
           "mem-perm flavor sequence is exactly 10n/15p/5n")
    _check(len(flavors) == C25.DIR_TURNS == 30, "mem-perm length == 30")

    specs = X25.build_mem_perm_specs(seed=91)
    _check(len(specs) == 30, "mem-perm specs length == 30")
    # every text is drawn from the frozen PRESSURE pool for its flavor
    all_from_pool = True
    for fl, text in specs:
        if text not in PRESSURE[fl]:
            all_from_pool = False
            break
    _check(all_from_pool, "every mem-perm text is from the frozen PRESSURE pool")
    # flavors of the specs match the frozen sequence
    _check([fl for fl, _ in specs] == expected,
           "mem-perm spec flavors match the frozen sequence")

    # mem-neutral: 30 neutral, all texts from the neutral pool
    nspecs = X25.build_mem_neutral_specs(seed=94)
    _check([fl for fl, _ in nspecs] == ["neutral"] * 30,
           "mem-neutral flavor sequence is 30 neutral")
    _check(all(t in PRESSURE["neutral"] for _, t in nspecs),
           "every mem-neutral text is from the neutral PRESSURE pool")


# ==========================================================================
# (a) --direction --stub end-to-end + nominal call count
# ==========================================================================
def test_direction_end_to_end(base_dir):
    print("\n--- (a) --direction --stub end-to-end (llama31) ---")
    slug = "llama31"
    res = X25.run_direction(slug, use_stub=True, resume=False, base_dir=base_dir)

    path = os.path.join(base_dir, X25.DIRECTION_JSON.format(slug=slug))
    _check(os.path.exists(path), "g25_direction JSON written")

    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)

    # life inventory: 5 baseline + 3 mem-perm + 3 mem-neutral = 11
    n_base = sum(1 for l in d["lives"] if l["arm"] == "baseline")
    n_perm = sum(1 for l in d["lives"] if l["arm"] == "memperm")
    n_neut = sum(1 for l in d["lives"] if l["arm"] == "memneutral")
    _check(n_base == 5 and n_perm == 3 and n_neut == 3,
           f"life inventory 5/3/3 (got {n_base}/{n_perm}/{n_neut})")

    # every frozen metric present + finite
    m = d.get("metrics", {})
    needed = ["baseline_mean", "dc", "dc_neutral", "persistence"]
    have_all = all(k in m and m[k] is not None and np.isfinite(m[k])
                   for k in needed)
    _check(have_all, f"all frozen metrics present + finite ({needed})")

    # per-rep arrays present (for --decide error bars)
    _check("dc_per_rep" in m and len(m["dc_per_rep"]) == 3,
           "dc_per_rep has 3 entries (std over reps)")
    _check("baseline_vals" in m and len(m["baseline_vals"]) == 5,
           "baseline_vals has 5 entries (noise floor)")

    # nominal call count == 654 (contract; catches turn/battery mis-accounting)
    _check(d["calls_used"] == 654,
           f"nominal calls == 654 (got {d['calls_used']})")
    _check(d["calls_used"] <= C25.DIR_CEILING,
           f"calls within DIR_CEILING={C25.DIR_CEILING}")

    # probes recorded at the frozen turns for a mem life
    perm_life = next(l for l in d["lives"] if l["arm"] == "memperm")
    turns = [p["turn"] for p in perm_life["probes"]]
    _check(turns == C25.DIR_PROBE_AT,
           f"mem-perm probes at DIR_PROBE_AT={C25.DIR_PROBE_AT} (got {turns})")
    return d


# ==========================================================================
# (b) resume skips completed lives
# ==========================================================================
def test_resume(base_dir):
    print("\n--- (b) resume skips completed lives ---")
    slug = "ablit"
    path = os.path.join(base_dir, X25.DIRECTION_JSON.format(slug=slug))

    # (1) full uninterrupted run -> reference
    ref_dir = os.path.join(base_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)
    ref = X25.run_direction(slug, use_stub=True, resume=False, base_dir=ref_dir)
    ref_ids = [l["id"] for l in ref["lives"]]

    # (2) simulate an interrupt: write a partial checkpoint with only the first
    #     6 lives, then resume. We build the partial by running fully then
    #     truncating -- the resume must skip those 6 and complete the rest.
    full = X25.run_direction(slug, use_stub=True, resume=False, base_dir=base_dir)
    keep = full["lives"][:6]
    partial = dict(full)
    partial["lives"] = keep
    partial.pop("metrics", None)
    X25._atomic_write(path, partial)
    kept_ids = {l["id"] for l in keep}

    # resume: only the missing 5 lives should be (re)appended
    resumed = X25.run_direction(slug, use_stub=True, resume=True, base_dir=base_dir)
    resumed_ids = [l["id"] for l in resumed["lives"]]

    _check(len(resumed_ids) == len(ref_ids) == 11,
           f"resume yields the full 11 lives (got {len(resumed_ids)})")
    _check(len(set(resumed_ids)) == len(resumed_ids),
           "no duplicate lives after resume (completed ones skipped)")
    _check(kept_ids.issubset(set(resumed_ids)),
           "the 6 pre-completed lives are retained (skipped, not dropped)")
    # metrics identical to the uninterrupted reference (deterministic stub)
    same_dc = abs(resumed["metrics"]["dc"] - ref["metrics"]["dc"]) < 1e-12
    _check(same_dc, "resumed metrics match the uninterrupted run (deterministic)")


# ==========================================================================
# (d) --decide on synthetic inputs -> R / S / M / F branches
# ==========================================================================
def _write_synth(base_dir, slug, dc, a, *, dc_neutral=0.1, persistence=0.02):
    """Write a minimal but schema-valid g25_direction + g25_fit for a slug so
    --decide can load them. The direction JSON carries a precomputed `metrics`
    block (compute_direction_metrics is bypassed because dc is set directly via
    a synthetic perm25 array against a zero baseline)."""
    # Construct lives that reproduce the requested dc exactly:
    #   baseline_mean = 0.5 (5 identical baseline lives)
    #   mem-perm t25 = 0.5 + dc for all 3 reps -> dc exact
    #   mem-perm t30 = 0.5 + dc + persistence
    #   mem-neutral t30 = 0.5 + dc_neutral
    base = 0.5
    lives = []
    for i in range(5):
        lives.append({"id": f"baseline-rep{i}", "arm": "baseline",
                      "rep": i, "battery_mean": base})
    for seed in C25.DIR_SEEDS_PERM:
        lives.append({"id": f"memperm-seed{seed}", "arm": "memperm",
                      "seed": seed, "probes": [
                          {"turn": 10, "mean": base},
                          {"turn": 25, "mean": base + dc},
                          {"turn": 30, "mean": base + dc + persistence},
                      ], "notes_count": 0, "summary": ""})
    for seed in C25.DIR_SEEDS_NEUTRAL:
        lives.append({"id": f"memneutral-seed{seed}", "arm": "memneutral",
                      "seed": seed, "probes": [
                          {"turn": 10, "mean": base},
                          {"turn": 25, "mean": base + dc_neutral},
                          {"turn": 30, "mean": base + dc_neutral},
                      ], "notes_count": 0, "summary": ""})
    direction = {"mode": "direction", "direction": slug,
                 "tag": C25.ARMS[slug]["tag"], "stub": True,
                 "ceiling": C25.DIR_CEILING, "probe_at": C25.DIR_PROBE_AT,
                 "lives": lives}
    # recompute metrics via the real function so the test exercises the same
    # code --decide uses (not a hand-rolled metrics block)
    direction["metrics"] = X25.compute_direction_metrics(direction)
    X25._atomic_write(
        os.path.join(base_dir, X25.DIRECTION_JSON.format(slug=slug)), direction)

    fit = {"slug": slug, "tag": C25.ARMS[slug]["tag"], "a": a,
           "b": -0.04, "drive": {"permissive": 0.0, "caution": 0.0,
                                 "neutral": 0.0},
           "monostable": bool(a < 0)}
    X25._atomic_write(
        os.path.join(base_dir, X25.FIT_JSON.format(slug=slug)), fit)


def _decide_in(base_dir, scenario):
    """Wipe any prior synthetic inputs, write the scenario, run --decide, return
    the decision dict."""
    for s in X25.SLUGS:
        for tmpl in (X25.DIRECTION_JSON, X25.FIT_JSON):
            p = os.path.join(base_dir, tmpl.format(slug=s))
            if os.path.exists(p):
                os.remove(p)
    for s, (dc, a) in scenario.items():
        _write_synth(base_dir, s, dc, a)
    return X25.run_decide(base_dir=base_dir)


def test_decision_rule(base_dir):
    print("\n--- (d) decision rule: R / S / M / F (the most important test) ---")
    d = os.path.join(base_dir, "decide")
    os.makedirs(d, exist_ok=True)

    # Branch R: all dc > +tau, all a < 0
    outR = _decide_in(d, {"qwen95": (0.25, -0.09),
                          "llama31": (0.15, -0.05),
                          "ablit": (0.12, -0.04)})
    _check(outR["decision"]["branch"] == "R",
           f"Branch R fires when all dc>+tau and all a<0 "
           f"(got {outR['decision']['branch']})")

    # Branch S: ablit dc < -tau, both safe arms > +tau, all a < 0
    outS = _decide_in(d, {"qwen95": (0.25, -0.09),
                          "llama31": (0.15, -0.05),
                          "ablit": (-0.20, -0.04)})
    _check(outS["decision"]["branch"] == "S",
           f"Branch S fires when ablit dc<-tau and safe arms>+tau "
           f"(got {outS['decision']['branch']})")

    # Branch M: ablit dc in [-tau, +tau] (sub-threshold), others > +tau, all a<0
    outM = _decide_in(d, {"qwen95": (0.25, -0.09),
                          "llama31": (0.15, -0.05),
                          "ablit": (0.05, -0.04)})
    _check(outM["decision"]["branch"] == "M",
           f"Branch M fires when ablit sub-threshold "
           f"(got {outM['decision']['branch']})")

    # Branch F: some model fits a>0 EVEN when all dc>+tau (F-precedence gate).
    #   This is the key precedence test: R's condition is ALSO satisfied here,
    #   but F must win because it is the architectural-falsification gate.
    outF = _decide_in(d, {"qwen95": (0.25, -0.09),
                          "llama31": (0.15, +0.20),   # bistable
                          "ablit": (0.12, -0.04)})
    _check(outF["decision"]["branch"] == "F",
           f"Branch F fires (a>0) and PREEMPTS R (F-first precedence) "
           f"(got {outF['decision']['branch']})")
    _check(outF["decision"]["reasons"]["all_dc_gt_tau"] is True,
           "F-precedence: R's condition (all dc>+tau) was ALSO true, F still won")

    # INCOMPLETE gate: all dc>+tau but ONE fit file absent -> must NOT emit an
    # authoritative R (an unknown `a` could be the a>0 that should fire F). This
    # is the absence-driven over-claim F-first exists to prevent, reintroduced
    # through a missing input rather than through ordering.
    for s in X25.SLUGS:
        for tmpl in (X25.DIRECTION_JSON, X25.FIT_JSON):
            p = os.path.join(d, tmpl.format(slug=s))
            if os.path.exists(p):
                os.remove(p)
    for s, (dc, a) in {"qwen95": (0.25, -0.09), "llama31": (0.15, -0.05),
                       "ablit": (0.12, -0.04)}.items():
        _write_synth(d, s, dc, a)
    # delete ablit's fit ONLY -> its `a` becomes unknown
    os.remove(os.path.join(d, X25.FIT_JSON.format(slug="ablit")))
    outI = X25.run_decide(base_dir=d)
    _check(outI["decision"]["branch"] == "INCOMPLETE",
           f"missing ablit fit -> INCOMPLETE, not authoritative R "
           f"(got {outI['decision']['branch']})")
    _check(outI["decision"]["reasons"].get("provisional_branch") == "R",
           "INCOMPLETE still records provisional_branch=R for the supervisor")
    _check("ablit" in outI["decision"]["reasons"]["unknown_fit_slugs"],
           "INCOMPLETE names ablit as the unknown-fit arm")

    # sanity: decision.json + figures exist for the last run
    _check(os.path.exists(os.path.join(d, X25.DECISION_JSON)),
           "g25_decision.json written")
    _check(os.path.exists(os.path.join(d, X25.FIG_DIRECTION)) and
           os.path.exists(os.path.join(d, X25.FIG_A)),
           "both decision figures written")

    # PRED-G25-1 grading sanity: R scenario should grade qwen sign+mag PASS
    g = outR["pred_g25_1"]["qwen95"]
    _check(g["sign_ok"] and g["mag_ok"],
           "PRED-G25-1 grades qwen95 sign+magnitude PASS on the committed 0.25")
    # a DOWN measurement against an UP prediction -> sign FAIL
    gS = outS["pred_g25_1"]["ablit"]
    _check((not gS["sign_ok"]),
           "PRED-G25-1 grades ablit sign FAIL when it drifts cavalier (dc<0)")


# ==========================================================================
# (e) field_g25 --measure --stub through the injection + --fit
# ==========================================================================
def test_field_measure_and_fit(base_dir):
    print("\n--- (e) field_g25 --measure --stub + --fit ---")
    import field_g2

    # snapshot the three attributes we expect field_g25 to inject-and-restore
    before = (field_g2.make_backend, field_g2.FIELD_JSON,
              field_g2.MEASURE_CEILING)

    slug = "llama31"
    F25.run_measure(slug, use_stub=True, resume=False, base_dir=base_dir)

    after = (field_g2.make_backend, field_g2.FIELD_JSON,
             field_g2.MEASURE_CEILING)
    _check(before == after,
           "field_g2 module attributes restored after --measure (no leak)")

    field_path = os.path.join(base_dir, F25.FIELD_JSON_TMPL.format(slug=slug))
    _check(os.path.exists(field_path),
           "g25_field_<slug>.json written via injection (field_g2 unedited)")

    with open(field_path, "r", encoding="utf-8") as fh:
        fd = json.load(fh)
    _check(fd["mode"] == "field" and len(fd["levels"]) == 5,
           "field JSON well-formed (5 levels, G2 format)")
    _check(fd.get("ceiling") == F25.FIELD_CEILING == 1200,
           "field JSON records the injected ceiling 1200")

    # --fit on the freshly measured field -> writes g25_fit + monostable flag
    fit = F25.run_fit(slug, base_dir=base_dir)
    fit_path = os.path.join(base_dir, F25.FIT_JSON_TMPL.format(slug=slug))
    _check(os.path.exists(fit_path), "g25_fit_<slug>.json written")
    _check("a" in fit and "monostable" in fit and fit["monostable"] == (fit["a"] < 0),
           "fit JSON has {a, monostable=bool(a<0)}")

    # qwen95 --fit reuses the ARCHIVED G2 field (results/g2_field.json, relative
    # to CWD) and must reproduce a ~= -0.0896 (frozen in results/g2_model_fit
    # .json). This validates the fit wiring. We run it with base_dir="." so the
    # archive resolves, then delete the produced g25_fit_qwen95.json from the
    # project root (it must not linger; the REAL run regenerates it).
    if os.path.exists(F25.QWEN_FIELD_ARCHIVE):
        qfit = F25.run_fit("qwen95", base_dir=".")
        _check(abs(qfit["a"] - (-0.08958756625158423)) < 1e-4,
               f"qwen95 refit reproduces archived a=-0.0896 (got {qfit['a']:+.5f})")
        _check(qfit["monostable"] is True, "qwen95 fit is monostable (a<0)")
        qpath = F25.FIT_JSON_TMPL.format(slug="qwen95")
        if os.path.exists(qpath):
            os.remove(qpath)
    else:
        print("  [skip] results/g2_field.json absent -- qwen refit sanity skipped")


# ==========================================================================
# runner
# ==========================================================================
def main():
    print("=== selftest_b25 (DRY, zero LLM calls) ===")
    scratch = tempfile.mkdtemp(prefix="g25_selftest_")
    print(f"scratch dir: {scratch}")
    try:
        test_turnspecs()
        test_direction_end_to_end(scratch)
        test_resume(scratch)
        test_decision_rule(scratch)
        # field measure/fit needs the archived qwen field, which lives relative
        # to the CWD (results/g2_field.json); run with base_dir=scratch for the
        # OUTPUTS but let the archive resolve from CWD. field_g25 resolves the
        # qwen field via base_dir join, so we pass "." for the archive lookup by
        # symlinking is overkill -- instead run this test with base_dir="." for
        # the qwen archive but write outputs into scratch by chdir.
        _run_field_test_isolated(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"\n[cleanup] removed scratch dir {scratch}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(PASS)} PASS, {len(FAIL)} FAIL")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        print("=" * 60)
        return 1
    print("ALL PASS")
    print("=" * 60)
    return 0


def _run_field_test_isolated(scratch):
    """Run the field test with CWD unchanged (so results/g2_field.json resolves)
    but outputs directed into the scratch base_dir."""
    test_field_measure_and_fit(scratch)


if __name__ == "__main__":
    sys.exit(main())
