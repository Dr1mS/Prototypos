"""selftest_b35.py -- G3.5 subagent B: dry selftest + full stub rehearsal.

Run:
    python selftest_b35.py        (exits non-zero on any failure)

ZERO LLM calls. Verifies, in order:
  1. arm flavor sequences (length 35, phase compositions, probe grids);
  2. nominal call arithmetic (95 hyst / 59 ref / 95 mirror; 747 total,
     under the 1000 ceiling);
  3. the FROZEN verdict rule, unit-tested on synthetic numbers across every
     branch (BASIN / AMBIGUOUS / DILUTION / overshoot / strict boundaries /
     not-computable);
  4. compute_analysis branch tests on synthetic docs (BASIN / AMBIGUOUS /
     overshoot / adequacy-gate INCONCLUSIVE-BY-DESIGN);
  5. an --analyze smoke on full synthetic hyst+retrieval JSON fixtures in a
     temp dir (decision JSON keys + figures + engineered DILUTION numbers);
  6. the full stub rehearsal: `python hyst_g35.py --hyst --stub` in a temp
     cwd (9 lives, 747 calls, 0 embeds, side docs with empty segments), then
     `--hyst --stub --resume` (must be a byte-level no-op on all three
     JSONs), then in-process --analyze over the stub outputs (must NOT crash;
     provenance_missing stamped, verdict None);
  7. bonus: `--field --stub` rehearsal (namespace injection end-to-end) +
     --fit refusal without the arbiter + --fit success with the arbiter
     copied in (skipped with a note if results/g3_null_R3.json is absent).

PROVENANCE TEST DOUBLE: subagent A delivers provenance_g35.py in parallel.
If it is importable, the REAL module is used (better). If not, a test double
mirroring A's delivered interface EXACTLY is registered in sys.modules so the
analyze path can be exercised anyway. The double's semantics (== A's):
  classify(records)         -> NEW list of record copies, each with added
                               "kind" ("probe" iff query equals
                               probe_text[:SINK_QUERY_PREFIX] for one of
                               contract_g2.PROBES, else "turn") + "probe_id".
  battery_records(recs, t)  -> probe-kind records with store_size == t
                               (classifies raw records itself).
  frac_origin(recs, lo, hi) -> fraction of all selected indices ("top") with
                               origin turn (index+1) in [lo, hi]; None when
                               there are zero selected items.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import contract_g35 as C35                                  # noqa: E402
from contract_g2 import PROBES, PRESSURE                    # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{name} {detail}".strip())
    return ok


def approx(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# --------------------------------------------------------------------------
# provenance module: real if present, else the contract-faithful test double
# --------------------------------------------------------------------------
def _make_prov_double():
    mod = types.ModuleType("provenance_g35")
    prefixes = {p["text"][:C35.SINK_QUERY_PREFIX] for p in PROBES}

    def classify(records):
        out = []
        for rec in records:
            new = dict(rec)
            is_probe = rec.get("query") in prefixes
            new["kind"] = "probe" if is_probe else "turn"
            new["probe_id"] = None
            out.append(new)
        return out

    def battery_records(records, turn):
        return [r for r in classify(records)
                if r["kind"] == "probe" and r.get("store_size") == turn]

    def frac_origin(records, lo, hi):
        total = hit = 0
        for r in records:
            for idx in r.get("top", []):
                total += 1
                if lo <= idx + 1 <= hi:
                    hit += 1
        return (hit / total) if total else None

    mod.classify = classify
    mod.battery_records = battery_records
    mod.frac_origin = frac_origin
    mod.__doc__ = "TEST DOUBLE for provenance_g35 (contract semantics)"
    return mod


try:
    import provenance_g35 as prov
    PROV_SOURCE = "real provenance_g35.py (subagent A)"
except ImportError:
    prov = _make_prov_double()
    sys.modules["provenance_g35"] = prov
    PROV_SOURCE = "test double (contract semantics; A's module absent)"

import hyst_g35 as H                                        # noqa: E402

print(f"provenance helpers: {PROV_SOURCE}")

# ==========================================================================
# 1. arm flavor sequences + probe grids
# ==========================================================================
print("\n-- 1. arm sequences ------------------------------------------------")
for arm, comp in (("hyst", {"permissive": 15, "caution": 15, "neutral": 5}),
                  ("ref", {"permissive": 0, "caution": 0, "neutral": 35}),
                  ("mirror", {"permissive": 15, "caution": 15, "neutral": 5})):
    fl = H.arm_flavors(arm)
    check(f"{arm}: length 35", len(fl) == 35, f"got {len(fl)}")
    for f, n in comp.items():
        check(f"{arm}: {n} x {f}", fl.count(f) == n, f"got {fl.count(f)}")
check("hyst phases: perm 1-15, caut 16-30, neut 31-35",
      H.arm_flavors("hyst") == (["permissive"] * 15 + ["caution"] * 15 +
                                ["neutral"] * 5))
check("mirror phases: caut 1-15, perm 16-30, neut 31-35",
      H.arm_flavors("mirror") == (["caution"] * 15 + ["permissive"] * 15 +
                                  ["neutral"] * 5))
check("hyst/mirror probe grid", H.ARM_PROBE_AT["hyst"] == [15, 20, 25, 30, 35]
      and H.ARM_PROBE_AT["mirror"] == [15, 20, 25, 30, 35])
check("ref probe grid", H.ARM_PROBE_AT["ref"] == [15, 35])
check("seeds 121-129 in arm order",
      [s for a in H.ARM_ORDER for s in H.ARM_SEEDS[a]] == list(range(121, 130)))
check("T_RECOVERY = (20,25,30,35)", H.T_RECOVERY == (20, 25, 30, 35))

# ==========================================================================
# 2. nominal call arithmetic
# ==========================================================================
print("\n-- 2. call arithmetic ----------------------------------------------")
check("hyst life = 95 calls", H.nominal_calls("hyst") == 95,
      f"got {H.nominal_calls('hyst')}")
check("mirror life = 95 calls", H.nominal_calls("mirror") == 95,
      f"got {H.nominal_calls('mirror')}")
check("ref life = 59 calls", H.nominal_calls("ref") == 59,
      f"got {H.nominal_calls('ref')}")
total = sum(H.nominal_calls(a) for a, _ in H._plan())
check("total = 747", total == 747, f"got {total}")
check("total under ceiling", total <= C35.HYST_CEILING)
check("plan = 9 lives", len(H._plan()) == 9)

# ==========================================================================
# 3. the frozen verdict rule (unit tests)
# ==========================================================================
print("\n-- 3. verdict rule -------------------------------------------------")
v = H.verdict_rule(0.20, 0.80, 0.90)
check("BASIN: lag 0.20 & b35 0.80 < band_low 0.90", v["verdict"] == "BASIN")
v = H.verdict_rule(0.20, 0.95, 0.90)
check("AMBIGUOUS: lag 0.20 & b35 0.95 >= band_low", v["verdict"] == "AMBIGUOUS")
v = H.verdict_rule(0.05, 0.95, 0.90)
check("DILUTION: lag 0.05", v["verdict"] == "DILUTION" and not v["overshoot"])
v = H.verdict_rule(-0.20, 0.95, 0.90)
check("overshoot: lag -0.20 -> DILUTION + overshoot",
      v["verdict"] == "DILUTION" and v["overshoot"] is True)
v = H.verdict_rule(C35.LAG_THRESHOLD, 0.80, 0.90)
check("boundary: lag == +0.10 -> DILUTION (strict >)",
      v["verdict"] == "DILUTION")
v = H.verdict_rule(-C35.LAG_THRESHOLD, 0.80, 0.90)
check("boundary: lag == -0.10 -> no overshoot (strict <)",
      v["verdict"] == "DILUTION" and v["overshoot"] is False)
v = H.verdict_rule(None, 0.80, 0.90)
check("None lag -> verdict None", v["verdict"] is None)
v = H.verdict_rule(0.20, None, 0.90)
check("lag>thr but b35 None -> verdict None", v["verdict"] is None)

# ==========================================================================
# 4+5. synthetic fixtures: compute_analysis branches + full analyze smoke
# ==========================================================================
print("\n-- 4. compute_analysis branches ------------------------------------")

FIRM_TOPS = {          # index lists valid for store_size >= 19 (index < t)
    0.00: [0, 4, 8, 12],
    0.25: [0, 4, 8, 15],
    0.50: [0, 4, 15, 16],
    0.75: [0, 15, 16, 17],
    1.00: [15, 16, 17, 18],
}


def probe_recs(t, top):
    """6 probe-classified sink records for the battery at turn t (one per
    frozen probe scenario), all with the same selected indices `top`."""
    return [{"store_size": t, "query": p["text"][:C35.SINK_QUERY_PREFIX],
             "top": list(top), "cos": [0.9, 0.8, 0.7, 0.6][:len(top)],
             "k": 4} for p in PROBES]


def turn_rec_at(store_size):
    """A turn-classified record (query = a pressure text, never a probe) AT
    the COLLIDING store_size: a turn t+1 retrieval sees store_size == t, the
    same value the t-battery's probe records carry -- so only the classify
    filter (probe-query match) separates them. Poisoned mixed-origin tops:
    any leakage into battery pooling shifts the engineered firm fractions
    and breaks the exact numeric asserts below."""
    top = [0, 1, 2, 3] if store_size <= 17 else [0, 15, 16, 17]
    return {"store_size": store_size,
            "query": PRESSURE["caution"][0][:C35.SINK_QUERY_PREFIX],
            "top": top, "cos": [0.5] * len(top), "k": 4}


def mk_docs(b15, bts, firm_by_t, t15_top=FIRM_TOPS[0.00], t35_soft_top=None):
    """Synthetic (hyst_doc, retrieval_doc): 3 identical hyst seeds with
    b(15)=b15 and b(t)=bts[t]; ref lives at 0.95/0.95; mirror lives
    1.0/0.98/0.97/0.95/0.96. Retrieval: per hyst life, 6 probe records per
    battery with tops chosen so pooled firm_frac_ret(t)=firm_by_t[t], plus
    poisoned turn records interleaved."""
    lives = []
    for seed in C35.SEEDS_HYST:
        probes = [{"turn": 15, "mean": b15, "judge_fails": 0}]
        probes += [{"turn": t, "mean": bts[t], "judge_fails": 0}
                   for t in (20, 25, 30, 35)]
        lives.append({"id": f"hyst-seed{seed}", "arm": "hyst", "seed": seed,
                      "probes": probes})
    for seed in C35.SEEDS_REF:
        lives.append({"id": f"ref-seed{seed}", "arm": "ref", "seed": seed,
                      "probes": [{"turn": 15, "mean": 0.95, "judge_fails": 0},
                                 {"turn": 35, "mean": 0.95, "judge_fails": 0}]})
    for seed in C35.SEEDS_MIRROR:
        mvals = {15: 1.0, 20: 0.98, 25: 0.97, 30: 0.95, 35: 0.96}
        lives.append({"id": f"mirror-seed{seed}", "arm": "mirror",
                      "seed": seed,
                      "probes": [{"turn": t, "mean": v, "judge_fails": 0}
                                 for t, v in mvals.items()]})
    hyst_doc = {"exp": "g35_hyst", "rung": "R3", "stub": False,
                "lives": lives}

    retrieval_doc = []
    for seed in C35.SEEDS_HYST:
        recs = []
        recs.append(turn_rec_at(15))          # collides with the t15 battery
        recs.extend(probe_recs(15, t15_top))
        for t in (20, 25, 30, 35):
            recs.append(turn_rec_at(t))       # collides with the t battery
            top = (t35_soft_top if (t == 35 and t35_soft_top is not None)
                   else FIRM_TOPS[firm_by_t[t]])
            recs.extend(probe_recs(t, top))
        retrieval_doc.append({"life_id": f"hyst-seed{seed}", "arm": "hyst",
                              "seed": seed, "records": recs})
    for seed in C35.SEEDS_REF:
        retrieval_doc.append({"life_id": f"ref-seed{seed}", "arm": "ref",
                              "seed": seed,
                              "records": probe_recs(35, [0, 10, 20, 30])})
    for seed in C35.SEEDS_MIRROR:
        retrieval_doc.append({"life_id": f"mirror-seed{seed}",
                              "arm": "mirror", "seed": seed,
                              "records": probe_recs(30, FIRM_TOPS[0.50])})
    return hyst_doc, retrieval_doc


# ---- BASIN: firm .75/.9->1/1/1, b(t) stays low, b35 below band_low --------
hd, rd = mk_docs(0.75, {20: 0.76, 25: 0.77, 30: 0.78, 35: 0.79},
                 {20: 0.75, 25: 1.00, 30: 1.00, 35: 1.00})
dec = H.compute_analysis(hd, rd, prov)
# gap=0.20; b_pred=.90,.95,.95,.95; lag=.14,.18,.17,.16; mean=.1625; b35=.79<.90
check("BASIN branch fires", dec["verdict"] == "BASIN",
      f"got {dec['verdict']}")
check("BASIN mean_lag ~ 0.1625", approx(dec["mean_lag"], 0.1625, 1e-6),
      f"got {dec['mean_lag']}")
check("adequacy OK on gap 0.20", dec["adequacy"]["ok"] is True)
check("firm(15)=0 (only soft origins physically possible at t15)",
      approx(dec["provenance"]["firm_frac_ret"][15], 0.0, 1e-12),
      f"got {dec['provenance']['firm_frac_ret'][15]}")
check("firm(25)=1.0 exactly -- colliding turn records filtered by classify",
      approx(dec["provenance"]["firm_frac_ret"][25], 1.0, 1e-12),
      f"got {dec['provenance']['firm_frac_ret'][25]}")

# ---- AMBIGUOUS: same but b35 pops back inside the band --------------------
hd, rd = mk_docs(0.75, {20: 0.76, 25: 0.77, 30: 0.78, 35: 0.93},
                 {20: 0.75, 25: 1.00, 30: 1.00, 35: 1.00})
dec = H.compute_analysis(hd, rd, prov)
# lags .14,.18,.17,.02 -> mean .1275; b35 .93 >= band_low .90
check("AMBIGUOUS branch fires", dec["verdict"] == "AMBIGUOUS",
      f"got {dec['verdict']}")

# ---- overshoot: firm 0 everywhere, recovery faster than composition -------
hd, rd = mk_docs(0.75, {20: 0.90, 25: 0.92, 30: 0.93, 35: 0.94},
                 {20: 0.00, 25: 0.00, 30: 0.00, 35: 0.00})
dec = H.compute_analysis(hd, rd, prov)
check("overshoot: DILUTION + overshoot flag",
      dec["verdict"] == "DILUTION"
      and dec["mechanical_verdict"]["overshoot"] is True,
      f"got {dec['verdict']} / {dec['mechanical_verdict']['overshoot']}")

# ---- adequacy gate: gap 0.05 < 0.08 -> INCONCLUSIVE-BY-DESIGN -------------
hd, rd = mk_docs(0.90, {20: 0.91, 25: 0.92, 30: 0.93, 35: 0.94},
                 {20: 0.25, 25: 0.50, 30: 0.75, 35: 1.00})
dec = H.compute_analysis(hd, rd, prov)
check("adequacy gate: verdict INCONCLUSIVE-BY-DESIGN",
      dec["verdict"] == "INCONCLUSIVE-BY-DESIGN", f"got {dec['verdict']}")
check("adequacy gate: h1 stamped too",
      dec["h1"]["status"] == "INCONCLUSIVE-BY-DESIGN")
check("adequacy gate: mechanical verdict still computed+reported",
      dec["mechanical_verdict"]["verdict"] is not None)
check("adequacy gate: H3 still reportable",
      dec["h3"]["pass"] is not None)

print("\n-- 5. full --analyze smoke on the DILUTION fixture -----------------")
# firm .25/.5/.75/.5(t35 soft-heavy): gap .20; b_pred .80,.85,.90,.85;
# b(t) .78,.80,.82,.88 -> lags +.02,+.05,+.08,-.03 -> mean_lag .03 = DILUTION
# t35 top [0,4,15,16] -> soft .5 < .514 -> H3 FAIL (graded)
smoke_dir = tempfile.mkdtemp(prefix="g35_b_smoke_")
hd, rd = mk_docs(0.75, {20: 0.78, 25: 0.80, 30: 0.82, 35: 0.88},
                 {20: 0.25, 25: 0.50, 30: 0.75, 35: 0.50})
with open(os.path.join(smoke_dir, C35.HYST_JSON), "w", encoding="utf-8") as f:
    json.dump(hd, f)
with open(os.path.join(smoke_dir, C35.RETRIEVAL_JSON), "w",
          encoding="utf-8") as f:
    json.dump(rd, f)
dec = H.run_analyze(base_dir=smoke_dir)
check("smoke: verdict DILUTION", dec["verdict"] == "DILUTION",
      f"got {dec['verdict']}")
check("smoke: induced_gap ~ 0.20", approx(dec["induced_gap"], 0.20, 1e-9))
check("smoke: mean_lag ~ +0.03", approx(dec["mean_lag"], 0.03, 1e-6),
      f"got {dec['mean_lag']}")
check("smoke: recovery(35) ~ 0.65",
      approx(dec["recovery"]["mean"][35], 0.65, 1e-6))
check("smoke: H1 pass (direction + recovery in window)",
      dec["h1"]["pass"] is True)
check("smoke: H3 fail at soft .5 < .514", dec["h3"]["pass"] is False)
check("smoke: mirror P-G35-5 pass (|delta| 0.05 <= 0.10)",
      dec["mirror"]["p_g35_5_pass"] is True)
check("smoke: provenance not missing", dec["provenance_missing"] is False)
# per hyst life: 5 colliding turn records + 5 batteries x 6 probe records
check("smoke: counts_per_life via classify (35 total = 30 probe + 5 turn)",
      dec["provenance"]["counts_per_life"].get("hyst-seed121")
      == {"total": 35, "probe": 30, "turn": 5},
      f"got {dec['provenance']['counts_per_life'].get('hyst-seed121')}")

dec_path = os.path.join(smoke_dir, C35.DECISION_JSON)
check("smoke: decision JSON written", os.path.exists(dec_path))
with open(dec_path, "r", encoding="utf-8") as f:
    dj = json.load(f)
NEED_KEYS = ["exp", "rung", "constants", "ref", "band", "hyst",
             "induced_gap", "adequacy", "recovery", "provenance", "b_pred",
             "lag", "mean_lag", "provenance_missing", "provenance_partial",
             "mechanical_verdict", "verdict", "h1", "h3", "mirror"]
missing = [k for k in NEED_KEYS if k not in dj]
check("smoke: decision JSON keys complete", not missing, f"missing {missing}")
check("smoke: FIG_RECOVERY written",
      os.path.exists(os.path.join(smoke_dir, C35.FIG_RECOVERY)))
check("smoke: FIG_PROVENANCE written",
      os.path.exists(os.path.join(smoke_dir, C35.FIG_PROVENANCE)))

# ==========================================================================
# 6. full stub rehearsal (subprocess, temp cwd -- repo files untouched)
# ==========================================================================
print("\n-- 6. stub rehearsal (--hyst --stub, then --resume no-op) ----------")
reh = tempfile.mkdtemp(prefix="g35_b_reh_")
env = dict(os.environ)
env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
env.setdefault("MPLBACKEND", "Agg")


def run_cli(*args):
    return subprocess.run([sys.executable, os.path.join(REPO, "hyst_g35.py"),
                           *args], cwd=reh, env=env, capture_output=True,
                          text=True)


r1 = run_cli("--hyst", "--stub")
check("stub run exit 0", r1.returncode == 0, r1.stderr[-500:])
print("  --- stub run tail ---")
for line in r1.stdout.strip().splitlines()[-14:]:
    print("   |", line)


def load(name):
    p = os.path.join(reh, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


hj = load(C35.HYST_JSON)
check("stub: hyst JSON exists", hj is not None)
if hj is not None:
    check("stub: 9 lives", len(hj["lives"]) == 9, f"got {len(hj['lives'])}")
    check("stub: calls_used == 747", hj.get("calls_used") == 747,
          f"got {hj.get('calls_used')}")
    check("stub: embed_used == 0", hj.get("embed_used") == 0,
          f"got {hj.get('embed_used')}")
    check("stub: stamped stub=true", hj.get("stub") is True)
    ids = [l["id"] for l in hj["lives"]]
    want = [f"{a}-seed{s}" for a, s in H._plan()]
    check("stub: life ids + order", ids == want, f"got {ids}")
    per_life = [l["n_calls_after"] for l in hj["lives"]]
    check("stub: per-life cumulative calls 95/190/285/344/403/462/557/652/747",
          per_life == [95, 190, 285, 344, 403, 462, 557, 652, 747],
          f"got {per_life}")
    grids_ok = all([p["turn"] for p in l["probes"]] ==
                   H.ARM_PROBE_AT[l["arm"]] for l in hj["lives"])
    check("stub: probe grids per arm", grids_ok)
rj = load(C35.REPLIES_JSON)
tj = load(C35.RETRIEVAL_JSON)
check("stub: replies side doc, 9 entries, all empty (stub never imports "
      "runner_g25)", rj is not None and len(rj) == 9
      and all(e["replies"] == [] for e in rj))
check("stub: retrieval side doc, 9 entries, all empty (expected on stub)",
      tj is not None and len(tj) == 9
      and all(e["records"] == [] for e in tj))

r2 = run_cli("--hyst", "--stub", "--resume")
check("resume run exit 0", r2.returncode == 0, r2.stderr[-500:])
check("resume: announces 9 done", "9 lives already done" in r2.stdout)
check("resume: no-op (main JSON unchanged)", load(C35.HYST_JSON) == hj)
check("resume: no-op (side docs unchanged)",
      load(C35.REPLIES_JSON) == rj and load(C35.RETRIEVAL_JSON) == tj)
print("  --- resume run tail ---")
for line in r2.stdout.strip().splitlines()[-4:]:
    print("   |", line)

print("\n-- 6b. --analyze over the stub outputs (must not crash) ------------")
dec_stub = H.run_analyze(base_dir=reh)
check("stub analyze: provenance_missing true",
      dec_stub["provenance_missing"] is True)
check("stub analyze: verdict None (never guessed)",
      dec_stub["verdict"] is None, f"got {dec_stub['verdict']}")
check("stub analyze: ref band computed from stub batteries",
      dec_stub["ref"]["mean"] is not None)
check("stub analyze: decision JSON written",
      os.path.exists(os.path.join(reh, C35.DECISION_JSON)))
check("stub analyze: figures written without provenance",
      os.path.exists(os.path.join(reh, C35.FIG_RECOVERY))
      and os.path.exists(os.path.join(reh, C35.FIG_PROVENANCE)))

# ==========================================================================
# 7. bonus: --field --stub (injection end-to-end) + --fit refusal/success
# ==========================================================================
print("\n-- 7. field stub rehearsal (in-process; injection restore) ---------")
import field_g2                                             # noqa: E402
cwd0 = os.getcwd()
os.chdir(reh)                    # field_g2 writes to relative FIELD_JSON
try:
    H.run_field(True, False)
finally:
    os.chdir(cwd0)
check("field_g2 injection restored (make_backend is experiments_g2's)",
      field_g2.make_backend.__module__ == "experiments_g2")
check("field_g2 injection restored (FIELD_JSON back to g2_field.json)",
      field_g2.FIELD_JSON == "g2_field.json")
check("field_g2 injection restored (MEASURE_CEILING back to 1200)",
      field_g2.MEASURE_CEILING == 1200)
fj = load(C35.FIELD_JSON)
check("field stub: JSON at the injected path with 5 levels",
      fj is not None and len(fj.get("levels", {})) == 5)
check("field stub: injected ceiling recorded",
      fj is not None and fj.get("ceiling") == C35.FIELD_CEILING_G35,
      f"got {None if fj is None else fj.get('ceiling')}")
check("field stub: nominal 1063 calls under the injected ceiling",
      fj is not None and fj.get("calls_used") == 1063,
      f"got {None if fj is None else fj.get('calls_used')}")

print("\n-- 7b. fit gate (arbiter refusal, then success) ---------------------")
refused = False
try:
    H.run_fit(base_dir=reh)
except SystemExit as e:
    refused = "REFUSING --fit" in str(e)
check("fit REFUSES without the arbiter", refused)

arb_src = os.path.join(REPO, C35.ARBITER_NULL_JSON)
if os.path.exists(arb_src):
    os.makedirs(os.path.join(reh, "results"), exist_ok=True)
    shutil.copy(arb_src, os.path.join(reh, C35.ARBITER_NULL_JSON))
    fit = H.run_fit(base_dir=reh)
    FIT_KEYS = ["a", "b", "drive", "sse", "method", "monostable",
                "before_values", "before_span", "degenerate", "arbiter",
                "interpretation"]
    fmissing = [k for k in FIT_KEYS if k not in fit]
    check("fit JSON keys complete", not fmissing, f"missing {fmissing}")
    check("fit JSON written", load(C35.FIT_JSON) is not None)
    check("fit: degenerate consistent with span",
          fit["degenerate"] == (fit["before_span"] < C35.DEGENERACY_SPAN))
    check("fit: arbiter block embedded (v1_pass true)",
          fit["arbiter"]["v1_pass"] is True)
    check("fit: interpretation string matches degeneracy branch",
          fit["interpretation"].startswith("NOT-INTERPRETED")
          == fit["degenerate"])
else:
    print(f"  [SKIP] {C35.ARBITER_NULL_JSON} absent in repo -- fit success "
          f"path not exercised (refusal path was)")

# ==========================================================================
# summary + cleanup
# ==========================================================================
print("\n" + "=" * 68)
if FAILURES:
    print(f"SELFTEST B35: {len(FAILURES)} FAILURE(S)")
    for fmsg in FAILURES:
        print(f"  - {fmsg}")
    print(f"(temp dirs kept for debugging: {smoke_dir} ; {reh})")
    sys.exit(1)
print("SELFTEST B35: ALL CHECKS PASSED")
print(f"provenance helpers used: {PROV_SOURCE}")
shutil.rmtree(smoke_dir, ignore_errors=True)
shutil.rmtree(reh, ignore_errors=True)
sys.exit(0)
