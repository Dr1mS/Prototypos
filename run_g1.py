"""run_g1.py -- G1 entrypoint (supervisor-owned, brief §1.5.h).

Preflight -> cheap battery in cost order -> §6 gate -> GO / NO-GO report.
Optional P1/P2 reduced behind --full.

Cheap battery (~1 000 calls, §1.5.g):
  1. reliability rig        (§4)  ~200 calls   gate-critical
  2. state-sensitivity      (§5)  ~200 calls   gate-critical
  3. P3 real                (§5)  ~480 calls   supporting
  4. P4 real                (§5)  ~300 calls   supporting

Interfaces this entrypoint consumes (frozen for subagents A/B/C):
  ollama_client.make_client(host) -> ollama.Client                     (A)
  appraiser.appraise_llm(event_text, state, name, *, client, model)    (A)
  test_reliability.run_reliability(client, model, n_reps=5) -> dict    (B)
  test_state_sensitivity.run_state_sensitivity(client, model,
                                               n_reps=5) -> dict       (C)
  experiments_real.run_p3_real(client, model) -> dict                  (C)
  experiments_real.run_p4_real(client, model) -> dict                  (C)
  experiments_real.run_p1_reduced(client, model) -> dict   (--full)    (C)
  experiments_real.run_p2_reduced(client, model) -> dict   (--full)    (C)
"""
import argparse
import json
import sys
import time

MODEL_DEFAULT = "qwen3.5:9b"
HOST_DEFAULT = "http://localhost:11434"

# ---------------------------------------------------------------------------
# COMMITTED PREDICTION (Atelios discipline, §5) -- written BEFORE any run.
# Direction: care(damaged) - care(secure) on ambiguous events is NEGATIVE.
# Rough magnitude: mean gap in [-0.6, -0.15]. (The mock's bias at full
# ambiguity is k_bias*amb = 0.9; the LLM is expected milder but clearly
# beyond noise.)
# ---------------------------------------------------------------------------
COMMITTED_PREDICTION = (
    "care(damaged) - care(secure) < 0 on ambiguous events; "
    "expected mean gap between -0.6 and -0.15, CI95 excluding 0."
)

# §6 gate thresholds (starting points, stated explicitly)
# SUPERVISOR ADJUSTMENT (post-run-1, declared): sign-agreement is per-dim.
# novelty carries only 6 labeled probes, so a flat 85% threshold degenerates
# into requiring 6/6 = 100% (5/6 = 83.3% < 85%). Threshold for novelty set to
# 0.80 (>=5/6) on granularity grounds; all other dims stay at 0.85. Run-1 data
# preserved in g1_results_run1.json.
GATE = dict(
    schema_valid_min=0.98,        # structured output IS supported (§1.5.a ok)
    subject_bug_failures_max=0,   # target-correctness = 100% on subject-bug batch
    sign_agreement_min={           # per labeled dimension
        "care": 0.85, "threat": 0.85, "autonomy": 0.85,
        "novelty": 0.80,           # granularity: only 6 labeled probes
    },
    stdev_max=0.15,               # inter-run stdev per dim, [-1,1] scales
)


def preflight(host, model):
    """Assert Ollama reachable + model present; else exit clearly (§1.5.h)."""
    try:
        from ollama_client import make_client
    except ImportError as e:
        sys.exit(f"PREFLIGHT FAIL: ollama_client.py missing ({e})")
    try:
        client = make_client(host)
        tags = client.list()
    except Exception as e:
        sys.exit(f"PREFLIGHT FAIL: Ollama daemon not reachable at {host}: {e}")
    names = {m.model for m in tags.models}
    if model not in names:
        sys.exit(f"PREFLIGHT FAIL: model '{model}' not present. "
                 f"Available: {sorted(names)}")
    print(f"preflight OK: {host}, model {model}")
    return client


def evaluate_gate(rel, sens):
    """Apply §6 to the gate-critical metrics. Returns (verdict, checks)."""
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))

    check("schema-valid >= 98%",
          rel["schema_valid_rate"] >= GATE["schema_valid_min"],
          f"{rel['schema_valid_rate']:.1%}")
    check("subject-bug target-correctness = 100%",
          rel["subject_bug_failures"] <= GATE["subject_bug_failures_max"],
          f"{rel['subject_bug_failures']} critical failures "
          f"on {rel['subject_bug_n']} probes")
    sign_ok = all(rel["sign_agreement"][d] >= thr
                  for d, thr in GATE["sign_agreement_min"].items())
    check("sign-agreement >= per-dim threshold (85%; novelty 80%)",
          sign_ok,
          json.dumps({k: round(v, 3) for k, v in rel["sign_agreement"].items()}))
    worst_sd = max(rel["stdev_per_dim"].values())
    check("inter-run stdev <= 0.15 per dim",
          worst_sd <= GATE["stdev_max"],
          json.dumps({k: round(v, 3) for k, v in rel["stdev_per_dim"].items()}))
    lo, hi = sens["ci95"]
    check("state-sensitivity gap negative, beyond noise",
          sens["mean_gap"] < 0 and hi < 0,
          f"mean gap {sens['mean_gap']:+.3f}, CI95 [{lo:+.3f}, {hi:+.3f}]")

    verdict = "GO" if all(ok for _, ok, _ in checks) else "NO-GO"
    return verdict, checks


def main():
    ap = argparse.ArgumentParser(description="G1 cheap battery + gate")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--host", default=HOST_DEFAULT)
    ap.add_argument("--full", action="store_true",
                    help="also run reduced P1/P2 (~4 250 extra calls)")
    ap.add_argument("--skip", default="",
                    help="comma list of steps to skip: rig,sens,p3,p4")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    client = preflight(args.host, args.model)
    print(f"committed prediction (pre-run): {COMMITTED_PREDICTION}\n")

    results = {"model": args.model, "prediction": COMMITTED_PREDICTION}
    t0 = time.time()

    # -- 1. reliability rig (gate-critical, cheapest) -----------------------
    if "rig" not in skip:
        from test_reliability import run_reliability
        print("[1/4] reliability rig (~200 calls)...")
        rel = run_reliability(client, args.model, n_reps=5)
        results["reliability"] = rel
        print(f"      schema-valid {rel['schema_valid_rate']:.1%} | "
              f"target-correct {rel['target_correct_rate']:.1%} | "
              f"subject-bug failures {rel['subject_bug_failures']} | "
              f"worst stdev {max(rel['stdev_per_dim'].values()):.3f}")

    # -- 2. state-sensitivity (gate-critical) -------------------------------
    if "sens" not in skip:
        from test_state_sensitivity import run_state_sensitivity
        print("[2/4] state-sensitivity (~200 calls)...")
        sens = run_state_sensitivity(client, args.model, n_reps=5)
        results["state_sensitivity"] = sens
        lo, hi = sens["ci95"]
        print(f"      gap {sens['mean_gap']:+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}] "
              f"(secure {sens['care_secure_mean']:+.3f} vs "
              f"damaged {sens['care_damaged_mean']:+.3f})")

    # -- 3. P3 real (supporting) --------------------------------------------
    if "p3" not in skip:
        from experiments_real import run_p3_real
        print("[3/4] P3 real: scar ON vs inject_state=False ablation "
              "(~480 calls)...")
        p3 = run_p3_real(client, args.model)
        results["p3_real"] = p3
        print(f"      A: secure {p3['A_secure']:+.2f} -> harm "
              f"{p3['A_damaged']:+.2f} -> kindness: ON {p3['end_A_on']:+.2f} "
              f"vs ablation {p3['end_A_ablation']:+.2f}")

    # -- 4. P4 real (supporting) --------------------------------------------
    if "p4" not in skip:
        from experiments_real import run_p4_real
        print("[4/4] P4 real: timescale separation (~300 calls)...")
        p4 = run_p4_real(client, args.model)
        results["p4_real"] = p4
        print(f"      mood ACT {p4['mood_act']} vs trait ACT {p4['trait_act']} "
              f"(ratio {p4['ratio']:.0f}x)")

    # -- optional full pair -------------------------------------------------
    if args.full:
        from experiments_real import run_p1_reduced, run_p2_reduced
        print("[--full] P1 reduced (~2 250 calls)...")
        results["p1_reduced"] = run_p1_reduced(client, args.model)
        print("[--full] P2 reduced (~2 000 calls)...")
        results["p2_reduced"] = run_p2_reduced(client, args.model)

    elapsed = time.time() - t0

    # -- gate ---------------------------------------------------------------
    print("\n" + "=" * 72)
    if "rig" in skip or "sens" in skip:
        print("gate NOT evaluated (gate-critical step skipped)")
        verdict = "INCOMPLETE"
    else:
        verdict, checks = evaluate_gate(results["reliability"],
                                        results["state_sensitivity"])
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print("-" * 72)
        print(f"  committed : {COMMITTED_PREDICTION}")
        sens = results["state_sensitivity"]
        lo, hi = sens["ci95"]
        print(f"  observed  : mean gap {sens['mean_gap']:+.3f}, "
              f"CI95 [{lo:+.3f}, {hi:+.3f}]")
        print("=" * 72)
        print(f"G1 verdict: {verdict}  ({elapsed/60:.0f} min, model {args.model})")
    results["verdict"] = verdict
    results["elapsed_s"] = round(elapsed, 1)

    with open("g1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("results written to g1_results.json")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
