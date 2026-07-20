"""experiments_field.py -- Subagent E (G1.5 §3). ZERO LLM calls.

Field-driven P1 (divergence) and P2 (path dependence) at the frozen prereg
scale, using `field_appraiser.make_field_appraiser` over subagent D's measured
`field_A.json`. Everything routes through the EXISTING `G0.run` dynamics -- no
re-implementation of the trait double-well (anti-duplication).

Reproducibility note (allowed runtime reseed, not a file edit): `G0.step` draws
its trait-noise from the module-global `G0.RNG` (seeded 7 at import). Our
`seed=` args only feed `make_stream`. So each run function reseeds `G0.RNG`
at entry, so `run_p1_field(seed=41)` / `run_p2_field(seed=42)` are deterministic
in isolation, independent of process call-order. `trait_noise=0.01` is tiny; the
verdict does not hinge on this, but the prereg names these seeds as reproducible.

P2 bug context (diagnosed, see report): the legacy alarming P2 figure
("first-quarter valence 1.00 for all points, r=0") was a STUB-ARTIFACT of
`experiments_real.run_p2_reduced`'s `base[:k]` prefix-subsample collapsing the
multiset to all-nurture at small scale. Production P2 at full scale is sound.
`run_p2_field` here NEVER subsamples the multiset -- full 114 always; only the
ordering count shrinks in the stub -- and asserts std(early_valence) > 0 so the
degeneracy cannot silently return.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

import G0
from field_appraiser import make_field_appraiser, validate_field

plt.style.use("dark_background")

RESULTS_JSON = "g15_field_results.json"

# G0.exp2's per-key valence map (verbatim; the early-quarter proxy).
_VALENCE = {"nurture": 1, "play": .6, "neutral": 0, "neglect": -.5,
            "scold": -.7, "harm": -1}

# G0's exact fixed multiset for P2 (prereg: 22+14+40+22+10+6 = 114).
_P2_MULTISET = (["nurture"] * 22 + ["play"] * 14 + ["neutral"] * 40 +
                ["neglect"] * 22 + ["scold"] * 10 + ["harm"] * 6)
assert len(_P2_MULTISET) == 114, len(_P2_MULTISET)


# --------------------------------------------------------------------------
# stream builder -- parameterized re-derivation of G0.biased_childhood with an
# INJECTED rng (G0's uses the module-global RNG so it can't be seeded per-run).
# Public: the supervisor reuses this for the §5 fidelity streams. Keeps the
# EXACT same pos/neg pools and warmth logic as G0.biased_childhood.
# --------------------------------------------------------------------------
def make_stream(rng, n, warmth):
    """Random life of `n` events with per-creature `warmth` bias in [0,1],
    using the injected `rng` (np.random.Generator). Same pools/logic as
    G0.biased_childhood."""
    pos = ["nurture", "play", "neutral"]
    neg = ["neglect", "scold", "harm", "neutral"]
    out = []
    for _ in range(n):
        out.append(rng.choice(pos) if rng.random() < warmth else rng.choice(neg))
    return out


# --------------------------------------------------------------------------
# P1 field-driven -- multiple attractors / divergence
# --------------------------------------------------------------------------
def run_p1_field(field_path="field_A.json", n_creatures=250, length=300,
                 seed=41, noise=True, fname="g15_fig_p1.png",
                 _field=None):
    """N creatures, each a make_stream(rng, length, warmth~U(0.15,0.85)) run
    through the field appraiser. Metrics: corner_fraction (|A|>0.6 & |R|>0.6),
    among cornered the damaged(A<0):secure(A>0) counts + ratio, mean_final_A,
    finals list. Figure: 2D scatter of finals (A,R) like G0.exp1.

    `_field` lets the stub-test pass a synthetic field dict directly instead of
    a path.
    """
    # reproducibility: reseed G0's trait-noise RNG (runtime, see module docstring)
    G0.RNG = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)

    field_src = _field if _field is not None else field_path
    appr = make_field_appraiser(field_src, noise=noise, seed=31)

    finals = []
    for _ in range(n_creatures):
        warmth = float(rng.uniform(0.15, 0.85))
        stream = make_stream(rng, length, warmth)
        traj = G0.run(stream, G0.P, appraiser=appr)
        finals.append(traj[-1, 2:4])  # (A, R)
    finals = np.asarray(finals)

    A = finals[:, 0]
    R = finals[:, 1]
    cornered_mask = (np.abs(A) > 0.6) & (np.abs(R) > 0.6)
    corner_fraction = float(np.mean(cornered_mask))

    corn_A = A[cornered_mask]
    damaged_n = int(np.sum(corn_A < 0))
    secure_n = int(np.sum(corn_A > 0))
    ratio = (float("inf") if secure_n == 0 and damaged_n > 0
             else (None if secure_n == 0 else damaged_n / secure_n))
    mean_final_A = float(np.mean(A))

    # figure (dark_background scatter; corner markers like G0.exp1)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(A, R, s=16, alpha=0.6, c="#00e5ff", edgecolors="none")
    for xx in (-1, 1):
        for yy in (-1, 1):
            ax.scatter([xx], [yy], marker="+", s=200, c="#ff2e88", lw=2)
    ax.axhline(0, color="#444", lw=0.7)
    ax.axvline(0, color="#444", lw=0.7)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel("A  (damaged  <->  secure)")
    ax.set_ylabel("R  (volatile  <->  serene)")
    ax.set_title("P1 field  %d creatures x %d events\ncorner-capture %.0f%%  "
                 "mean A=%.2f  (dmg:sec %d:%d)"
                 % (n_creatures, length, 100 * corner_fraction, mean_final_A,
                    damaged_n, secure_n), fontsize=10.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)

    return dict(corner_fraction=corner_fraction, damaged_n=damaged_n,
                secure_n=secure_n, damaged_secure_ratio=ratio,
                mean_final_A=mean_final_A, n_creatures=int(n_creatures),
                length=int(length), noise=bool(noise),
                finals=finals.tolist(), fig=fname)


# --------------------------------------------------------------------------
# P2 field-driven -- path dependence (order matters)
# --------------------------------------------------------------------------
def run_p2_field(field_path="field_A.json", n_orderings=40, seed=42,
                 noise=True, fname="g15_fig_p2.png", _field=None):
    """G0's EXACT 114-event multiset, `n_orderings` random orderings, final A
    per ordering. Metrics: spread_std, early_corr (r), early_slope. Early-quarter
    valence uses G0.exp2's per-key valence map over the actual permuted order.

    SANITY ASSERT: std(early_valence) > 0 across orderings -- a degenerate
    all-equal early-valence (the legacy bug) MUST raise. The multiset is NEVER
    subsampled; only `n_orderings` shrinks for the stub.
    """
    G0.RNG = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)

    field_src = _field if _field is not None else field_path
    appr = make_field_appraiser(field_src, noise=noise, seed=31)

    base = list(_P2_MULTISET)  # full 114, ALWAYS -- never subsampled

    finals, early_val = [], []
    for _ in range(n_orderings):
        order = list(rng.permutation(base))
        traj = G0.run(order, G0.P, appraiser=appr)
        finals.append(float(traj[-1, 2]))
        q = max(1, len(order) // 4)
        early_val.append(float(np.mean([_VALENCE[e] for e in order[:q]])))
    finals = np.asarray(finals)
    early_val = np.asarray(early_val)

    # THE bug guard: a correct P2 MUST show spread in first-quarter valence.
    assert float(early_val.std()) > 0.0, (
        "degenerate early-valence (all orderings equal) -- the P2 multiset "
        "collapsed; refusing to report a flat figure")

    spread_std = float(finals.std())
    slope, intercept = np.polyfit(early_val, finals, 1)
    corr = float(np.corrcoef(early_val, finals)[0, 1])

    # figure: histogram + scatter like G0.exp2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    ax[0].hist(finals, bins=min(24, max(4, n_orderings)), color="#00e5ff",
               alpha=0.85)
    ax[0].axvline(0, color="#ff2e88", lw=1.2)
    ax[0].set_xlabel("final attachment A")
    ax[0].set_ylabel("orderings")
    ax[0].set_title("identical 114-event multiset,\n%d random orderings"
                    % n_orderings, fontsize=11)
    ax[1].scatter(early_val, finals, s=24, alpha=0.6, c="#00e5ff",
                  edgecolors="none")
    xs = np.linspace(early_val.min(), early_val.max(), 20)
    ax[1].plot(xs, intercept + slope * xs, color="#ff2e88", lw=2)
    ax[1].set_xlabel("mean valence of FIRST quarter of life")
    ax[1].set_ylabel("final attachment A")
    ax[1].set_title("early events dominate\nr = %.2f  slope = %.2f"
                    % (corr, slope), fontsize=11)
    fig.suptitle("P2 field  path dependence: WHEN, not just how much",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)

    return dict(spread_std=spread_std, early_corr=corr,
                early_slope=float(slope), n_orderings=int(n_orderings),
                noise=bool(noise), finals=finals.tolist(),
                early_val=early_val.tolist(), fig=fname)


# --------------------------------------------------------------------------
# prereg PRED-1 check (informational -- verdict belongs to the supervisor)
# --------------------------------------------------------------------------
def _pred1_lines(p1):
    """Return human-readable PASS/FAIL lines vs PRED-1 thresholds. These are
    FYI; the supervisor owns the verdict (prereg: no moved pass-line)."""
    cf = p1["corner_fraction"]
    ratio = p1["damaged_secure_ratio"]
    mA = p1["mean_final_A"]
    a = cf < 0.50
    # ratio None (no secure) or inf both satisfy >= 2:1; numeric must be >= 2
    if ratio is None or ratio == float("inf"):
        b = p1["damaged_n"] > 0
        ratio_str = f"{p1['damaged_n']}:0"
    else:
        b = ratio >= 2.0
        ratio_str = f"{ratio:.2f}"
    c = mA < -0.15
    return [
        f"  PRED-1a corner-capture < 50%%: {100*cf:.0f}%%  -> "
        f"{'PASS' if a else 'FAIL'}",
        f"  PRED-1b damaged:secure >= 2:1: {ratio_str} "
        f"(dmg={p1['damaged_n']} sec={p1['secure_n']}) -> "
        f"{'PASS' if b else 'FAIL'}",
        f"  PRED-1c mean final A < -0.15: {mA:+.3f}  -> "
        f"{'PASS' if c else 'FAIL'}",
    ]


def _save_results(field_path, seeds, p1, p2):
    """Write the supervisor-consumed results JSON. numpy -> native handled by
    the run functions' .tolist()/float()."""
    out = {
        "field_path": field_path,
        "seeds": seeds,
        "p1": p1,
        "p2": p2,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return RESULTS_JSON


# --------------------------------------------------------------------------
# __main__: (1) stub-test both at tiny sizes; (2) IF field_A.json is present +
# complete, run BOTH at full frozen scale and save results.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from field_appraiser import _synthetic_field

    print("=== experiments_field self-test (stub field, no LLM) ===")
    stub_field = _synthetic_field()

    p1s = run_p1_field(_field=stub_field, n_creatures=8, length=30, seed=41,
                       noise=True, fname="g15_fig_p1.png")
    req1 = {"corner_fraction", "damaged_n", "secure_n", "damaged_secure_ratio",
            "mean_final_A", "finals", "fig"}
    assert req1 <= set(p1s), req1 - set(p1s)
    assert len(p1s["finals"]) == 8
    assert os.path.exists(p1s["fig"]) and p1s["fig"] == "g15_fig_p1.png"
    print(f"  P1 stub OK: corner={p1s['corner_fraction']:.2f} "
          f"meanA={p1s['mean_final_A']:+.2f} dmg={p1s['damaged_n']} "
          f"sec={p1s['secure_n']} finals={len(p1s['finals'])} fig={p1s['fig']}")

    p2s = run_p2_field(_field=stub_field, n_orderings=5, seed=42, noise=True,
                       fname="g15_fig_p2.png")
    req2 = {"spread_std", "early_corr", "early_slope", "finals", "fig"}
    assert req2 <= set(p2s), req2 - set(p2s)
    assert len(p2s["finals"]) == 5
    assert os.path.exists(p2s["fig"]) and p2s["fig"] == "g15_fig_p2.png"
    print(f"  P2 stub OK: spread_std={p2s['spread_std']:.3f} "
          f"early_corr={p2s['early_corr']:+.2f} "
          f"early_slope={p2s['early_slope']:+.2f} "
          f"finals={len(p2s['finals'])} fig={p2s['fig']}")
    print("=== stub self-test PASSED ===")

    # ---- full-scale block (only if D's field exists AND parses complete) ----
    FIELD = "field_A.json"
    run_full = False
    if os.path.exists(FIELD):
        try:
            validate_field(FIELD)
            run_full = True
        except Exception as e:
            print(f"[SKIPPED full-scale] {FIELD} present but incomplete: {e}")
    else:
        print(f"[SKIPPED full-scale] {FIELD} absent -- supervisor will run "
              f"this once subagent D delivers the field.")

    if run_full:
        print("\n=== full-scale field-driven run (prereg frozen params) ===")
        p1 = run_p1_field(field_path=FIELD, n_creatures=250, length=300,
                          seed=41, noise=True)
        print(f"P1: corner-capture={100*p1['corner_fraction']:.0f}%  "
              f"mean A={p1['mean_final_A']:+.3f}  "
              f"cornered dmg:sec={p1['damaged_n']}:{p1['secure_n']}")
        for line in _pred1_lines(p1):
            print(line)

        p2 = run_p2_field(field_path=FIELD, n_orderings=40, seed=42, noise=True)
        print(f"P2: spread_std={p2['spread_std']:.3f}  "
              f"early_corr={p2['early_corr']:+.2f}  "
              f"early_slope={p2['early_slope']:+.2f}")

        path = _save_results(FIELD, {"p1": 41, "p2": 42}, p1, p2)
        print(f"\nsaved -> {path}  (figs: {p1['fig']}, {p2['fig']})")
        print("NOTE: PRED-1 lines above are informational; the verdict belongs "
              "to the supervisor (prereg: no moved pass-line).")
