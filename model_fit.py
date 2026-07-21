"""model_fit.py -- G2 subagent C: the G0 model bridge (PRED-4). Zero LLM calls.

CLI:
    python model_fit.py

Fits the G0 1-D double-well dynamics to the MEASURED caution transition field
(g2_field.json), then simulates the Exp-1 and Exp-2 protocols with the fitted
model and evaluates PRED-G2-4a/4b against the OBSERVED values loaded from
g2_exp1_results.json / g2_exp2_results.json.

THE MODEL (G0's rule, mapped to caution -- prereg caution<->x affine map):
    x     = 2.4 * (caution - 0.5)          # caution [0,1] <-> x [-1.2, +1.2]
    dx    = (a*x - b*x^3 + drive_f)         # per-flavor drive_f, f in
                                            #   {permissive, caution, neutral}
    x'    = clip(x + dx, -1.2, 1.2)         # G0's clip range
    caution' = x'/2.4 + 0.5

IDENTIFIABILITY (critical): in G0 the rule is dx=(a*x - b*x^3 + drive)*lr, and
a, b, drive, lr appear only as PRODUCTS. lr is therefore NOT identifiable from a
single-step field. We PIN lr = 1.0 and fold it into (a, b, drive_*): 5 free
params [a, b, d_permissive, d_caution, d_neutral], fit by least squares on the
measured Delta field (15 points = 5 levels x 3 flavors). scipy.optimize is used
if available; else a numpy grid+refine fallback.

The Delta the field measured is in CAUTION units; we convert each (before,
after) to x-space and fit dx there, so the fitted a/b/drives live in G0's native
coordinates and the simulations reuse the exact same map.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

FIELD_JSON = "g2_field.json"
EXP1_JSON = "g2_exp1_results.json"
EXP2_JSON = "g2_exp2_results.json"
OUT_JSON = "g2_model_fit.json"
BRIDGE_FIG = "g2_fig_bridge.png"

FLAVORS = ["permissive", "caution", "neutral"]
X_CLIP = 1.2

# PRED-4 pass-lines (prereg -- FROZEN)
PRED_4A_RECOVERY_TOL = 0.25
PRED_4B_R_SIGN = True
PRED_4B_R_TOL = 0.35

# Exp-1 protocol geometry (mirrors experiments_g2, kept local to avoid a hard
# dependency loop; the ordering generator IS imported from experiments_g2).
EXP1_SEQ = (["neutral"] * 10 + ["permissive"] * 15 + ["caution"] * 15 +
            ["neutral"] * 10)
EXP1_PROBE_AT = [10, 15, 20, 25, 30, 35, 40, 45, 50]


# ==========================================================================
# caution <-> x map (prereg)
# ==========================================================================
def caution_to_x(c):
    return 2.4 * (np.asarray(c, float) - 0.5)


def x_to_caution(x):
    return np.asarray(x, float) / 2.4 + 0.5


# ==========================================================================
# load the measured field as (x_before, delta_x) points per flavor
# ==========================================================================
def load_field_points(path=FIELD_JSON):
    """Return {flavor: (x_before[array], dx[array])} from the measured field.
    dx is the change in x implied by the measured caution Delta at each level."""
    with open(path, "r", encoding="utf-8") as fh:
        field = json.load(fh)
    levels = field["levels"]
    pts = {f: {"x": [], "dx": [], "before_c": []} for f in FLAVORS}
    for lv, entry in levels.items():
        before_c = entry["before"]
        x_before = float(caution_to_x(before_c))
        for f in FLAVORS:
            mean_delta_c = entry["cells"][f]["mean_delta"]
            after_c = before_c + mean_delta_c
            dx = float(caution_to_x(after_c) - caution_to_x(before_c))
            pts[f]["x"].append(x_before)
            pts[f]["dx"].append(dx)
            pts[f]["before_c"].append(before_c)
    for f in FLAVORS:
        pts[f]["x"] = np.array(pts[f]["x"], float)
        pts[f]["dx"] = np.array(pts[f]["dx"], float)
        pts[f]["before_c"] = np.array(pts[f]["before_c"], float)
    return pts, field


# ==========================================================================
# fit  dx = a*x - b*x^3 + drive_f   (lr pinned to 1)
# ==========================================================================
def _residuals(params, pts):
    a, b, dp, dc, dn = params
    drives = {"permissive": dp, "caution": dc, "neutral": dn}
    res = []
    for f in FLAVORS:
        x = pts[f]["x"]
        pred = a * x - b * x ** 3 + drives[f]
        res.append(pred - pts[f]["dx"])
    return np.concatenate(res)


def fit_model(pts):
    """Least-squares fit of [a, b, d_perm, d_caut, d_neut]. Uses
    scipy.optimize.least_squares if available; else a numpy grid+refine."""
    x0 = np.array([0.85, 0.85, -0.3, 0.3, 0.0])
    try:
        from scipy.optimize import least_squares
        sol = least_squares(_residuals, x0, args=(pts,), method="lm",
                            max_nfev=20000)
        params = sol.x
        cost = float(np.sum(sol.fun ** 2))
        method = "scipy.least_squares"
    except Exception as e:  # numpy fallback (grid + local refine)
        params, cost = _numpy_fit(pts, x0)
        method = f"numpy grid+refine (scipy unavailable: {e})"
    a, b, dp, dc, dn = params
    return {"a": float(a), "b": float(b),
            "drive": {"permissive": float(dp), "caution": float(dc),
                      "neutral": float(dn)},
            "sse": cost, "method": method}


def _numpy_fit(pts, x0):
    """Coordinate grid+refine fallback: given (a,b) the drives are the per-flavor
    mean residual (linear closed form), so we grid over (a,b) and pick the best."""
    best = None
    for a in np.linspace(-0.5, 2.0, 60):
        for b in np.linspace(-0.5, 2.0, 60):
            drives = {}
            sse = 0.0
            for f in FLAVORS:
                x = pts[f]["x"]
                base = a * x - b * x ** 3
                d = float(np.mean(pts[f]["dx"] - base))   # optimal drive
                drives[f] = d
                sse += float(np.sum((base + d - pts[f]["dx"]) ** 2))
            if best is None or sse < best[1]:
                best = ([a, b, drives["permissive"], drives["caution"],
                         drives["neutral"]], sse)
    return np.array(best[0]), best[1]


# ==========================================================================
# simulate the fitted model
# ==========================================================================
def sim_step(x, flavor, model):
    a, b = model["a"], model["b"]
    drive = model["drive"][flavor]
    dx = a * x - b * x ** 3 + drive
    return float(np.clip(x + dx, -X_CLIP, X_CLIP))


def sim_sequence(sequence, model, start_c):
    """Run the fitted model over a flavor sequence, returning caution per turn
    (list) and the final caution."""
    x = float(caution_to_x(start_c))
    caut = []
    for fl in sequence:
        x = sim_step(x, fl, model)
        caut.append(float(x_to_caution(x)))
    return caut, (caut[-1] if caut else float(x_to_caution(x)))


def sim_exp1(model, start_c):
    """Simulate the Exp-1 protocol; return the recovery ratio per the prereg
    formula (baseline@10, min(25,30), final@50)."""
    caut, _ = sim_sequence(EXP1_SEQ, model, start_c)
    # caut is 1-based-by-position: caut[i] is caution AFTER turn i+1
    def at(turn):
        return caut[turn - 1]
    baseline = at(10)
    lo = min(at(25), at(30))
    drop = baseline - lo
    recovery = ((at(50) - lo) / drop) if abs(drop) > 1e-9 else float("nan")
    return {"baseline": baseline, "min": lo, "final": at(50),
            "drop": drop, "recovery_ratio": recovery}


def sim_exp2(model, start_c):
    """Simulate Exp-2's 12 seed-71 orderings (imported generator) with the fitted
    model; return (finals, fq_shares, std, pearson_r)."""
    from experiments_g2 import make_exp2_orderings, first_quarter_share
    orderings = make_exp2_orderings()
    finals, fq = [], []
    for order in orderings:
        _, final_c = sim_sequence(order, model, start_c)
        finals.append(final_c)
        fq.append(first_quarter_share(order))
    finals = np.array(finals, float)
    fq = np.array(fq, float)
    std = float(np.std(finals))
    if np.std(finals) > 0 and np.std(fq) > 0:
        r = float(np.corrcoef(fq, finals)[0, 1])
    else:
        r = float("nan")
    return finals, fq, std, r


# ==========================================================================
# observed values from the experiment results
# ==========================================================================
def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def observed_recovery(exp1):
    recs = [l["metrics"]["recovery_ratio"] for l in exp1["lives"]
            if l["arm"] == "memory" and l.get("metrics")
            and np.isfinite(l["metrics"]["recovery_ratio"])]
    return float(np.mean(recs)) if recs else float("nan")


def observed_r(exp2):
    from experiments_g2 import exp2_arm_stats
    _, _, _, r = exp2_arm_stats(exp2, "memory")
    return r


# ==========================================================================
# main
# ==========================================================================
def main():
    if not os.path.exists(FIELD_JSON):
        raise SystemExit(f"{FIELD_JSON} absent -- run field_g2.py --measure first.")
    pts, field = load_field_points(FIELD_JSON)
    model = fit_model(pts)

    print("=" * 66)
    print("MODEL FIT -- G0 double-well fitted to the measured caution field")
    print("=" * 66)
    print(f"  method: {model['method']}")
    print(f"  a={model['a']:+.4f}  b={model['b']:+.4f}  SSE={model['sse']:.5f}")
    print(f"  drives: permissive={model['drive']['permissive']:+.4f}  "
          f"caution={model['drive']['caution']:+.4f}  "
          f"neutral={model['drive']['neutral']:+.4f}")

    # start caution for the sims: the L3 (neutral-prefix) before value if
    # present, else 0.5 (x=0). Documented.
    start_c = 0.5
    if "L3" in field["levels"]:
        start_c = field["levels"]["L3"]["before"]
    # clamp to a sane baseline for the sim start (avoid starting at an extreme)
    print(f"  sim start caution = {start_c:.3f}")

    sim1 = sim_exp1(model, start_c)
    finals2, fq2, std2, r2_sim = sim_exp2(model, start_c)
    print(f"\n  simulated Exp-1 recovery ratio = {sim1['recovery_ratio']:.3f} "
          f"(drop {sim1['drop']:.3f})")
    print(f"  simulated Exp-2 first-quarter r = {r2_sim:+.3f} "
          f"(std {std2:.3f})")

    out = {"model": model, "sim_start_caution": start_c,
           "sim_exp1": sim1,
           "sim_exp2": {"std": std2, "r": r2_sim,
                        "finals": finals2.tolist(), "fq_shares": fq2.tolist()}}

    # ---- PRED-4 vs observed (if experiment results exist) ----
    exp1 = _load(EXP1_JSON)
    exp2 = _load(EXP2_JSON)
    print("\n" + "-" * 66)
    print("PRED-G2-4 (informational; verdict = supervisor)")
    print("-" * 66)
    if exp1 is not None:
        obs_rec = observed_recovery(exp1)
        diff = abs(sim1["recovery_ratio"] - obs_rec) if np.isfinite(
            sim1["recovery_ratio"]) and np.isfinite(obs_rec) else float("nan")
        p4a = np.isfinite(diff) and diff < PRED_4A_RECOVERY_TOL
        print(f"  observed Exp-1 recovery = {obs_rec:.3f} ; simulated = "
              f"{sim1['recovery_ratio']:.3f} ; |diff| = {diff:.3f}")
        print(f"  PRED-G2-4a |sim - obs recovery| < {PRED_4A_RECOVERY_TOL}: "
              f"-> {'PASS' if p4a else 'FAIL'}")
        out["pred4a"] = {"observed": obs_rec, "sim": sim1["recovery_ratio"],
                         "diff": diff, "pass": bool(p4a)}
    else:
        print(f"  {EXP1_JSON} absent -- run --exp1 to evaluate PRED-4a")
    if exp2 is not None:
        obs_r = observed_r(exp2)
        same_sign = (np.isfinite(obs_r) and np.isfinite(r2_sim)
                     and np.sign(obs_r) == np.sign(r2_sim) and obs_r != 0)
        rdiff = abs(r2_sim - obs_r) if np.isfinite(obs_r) and np.isfinite(
            r2_sim) else float("nan")
        p4b = same_sign and np.isfinite(rdiff) and rdiff < PRED_4B_R_TOL
        print(f"  observed Exp-2 r = {obs_r:+.3f} ; simulated = {r2_sim:+.3f} ; "
              f"|diff| = {rdiff:.3f} ; same sign = {same_sign}")
        print(f"  PRED-G2-4b same sign AND |r_sim - r_obs| < {PRED_4B_R_TOL}: "
              f"-> {'PASS' if p4b else 'FAIL'}")
        out["pred4b"] = {"observed": obs_r, "sim": r2_sim, "diff": rdiff,
                         "same_sign": bool(same_sign), "pass": bool(p4b)}
    else:
        print(f"  {EXP2_JSON} absent -- run --exp2 to evaluate PRED-4b")
    print("-" * 66)

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    _plot_bridge(pts, model)
    print(f"\nwritten: {OUT_JSON}, {BRIDGE_FIG}")
    return out


def _plot_bridge(pts, model):
    """Measured Delta-field points vs fitted dx curves, per flavor (caution
    units on the x-axis for readability)."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"permissive": "#ff2e88", "caution": "#00e5ff", "neutral": "#ffd166"}
    xs = np.linspace(-X_CLIP, X_CLIP, 100)
    for f in FLAVORS:
        # measured points
        c_before = pts[f]["before_c"]
        dx = pts[f]["dx"]
        ax.scatter(c_before, dx, s=60, c=colors[f], edgecolors="none",
                   label=f"{f} (measured)")
        # fitted curve
        pred = model["a"] * xs - model["b"] * xs ** 3 + model["drive"][f]
        ax.plot(x_to_caution(xs), pred, color=colors[f], lw=1.6, alpha=0.8)
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xlabel("current caution (before)")
    ax.set_ylabel("dx  (change in x = 2.4*(caution-0.5))")
    ax.set_title(f"G2 model bridge: fitted G0 double-well over measured field\n"
                 f"a={model['a']:.2f} b={model['b']:.2f} "
                 f"SSE={model['sse']:.4f}", fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.3)
    fig.tight_layout()
    fig.savefig(BRIDGE_FIG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
