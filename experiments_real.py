"""experiments_real.py -- G1 §5 reduced real-LLM experiments (subagent C).

These runs go THROUGH the harness `run()` with the real appraiser from
`make_real_appraiser` (seam.py). We never re-implement dynamics and never touch
p["k_bias"] (double-bias guard, §1.5.d). `bias_on` is meaningless on the real
path; the real "bias OFF" ablation is `inject_state=False` (§1.5.e). Gate runs
use text_pick="first".

Model: qwen3.5:9b, ~1.4 s/call. Every function returns `n_calls` and
`hard_fails` and saves a dark_background figure.

Hard-fail policy (subagent-C brief): appraise_llm raises AppraisalError after
its own internal retry. Aborting a 242-event run for one bad call is
unacceptable, so we wrap the appraiser with `_robust`: one extra retry, then a
counted neutral fallback (0.0, 0.0, 0.1) so the trajectory stays in sync (we
CANNOT drop a mid-stream event without desyncing `run()`). This differs from
test_state_sensitivity, which drops reps.

Lazy imports of the appraiser stack live inside functions so this module
imports even when appraiser.py / parse.py are absent (written in parallel).
"""

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")


# --------------------------------------------------------------------------
# robust appraiser wrapper (neutral fallback, counted) -- § hard-fail policy
# --------------------------------------------------------------------------
def _robust(appr, counter, label="", every=20):
    """Wrap a real appraiser: one extra retry on AppraisalError, then a counted
    neutral fallback so a single bad call cannot abort a long run.

    `counter` is a dict; we bump counter["n_calls"] once per logical event and
    counter["hard_fails"] when we fall back. Progress is printed every `every`
    events (brief: "print progress every ~20 events"). AppraisalError import is
    lazy and degrades to a sentinel if parse.py is absent.
    """
    try:
        from parse import AppraisalError
    except Exception:
        class AppraisalError(Exception):
            pass

    def wrapped(ev, s, age, p, bias_on):
        counter["n_calls"] += 1
        if every and counter["n_calls"] % every == 0:
            print(f"    {label}: {counter['n_calls']} events "
                  f"({counter['hard_fails']} hard-fails)")
        for attempt in (0, 1):
            try:
                return appr(ev, s, age, p, bias_on)
            except AppraisalError:
                if attempt == 1:
                    counter["hard_fails"] += 1
                    return (0.0, 0.0, 0.1)  # neutral fallback, counted
    return wrapped


def _make_counter():
    return dict(n_calls=0, hard_fails=0)


def _default_make_appraiser(*args, **kwargs):
    """Lazy default for `_make_appraiser` -> seam.make_real_appraiser.
    Kept as a thin wrapper so importing this module doesn't import seam (which
    would try to import appraiser at call time only)."""
    from seam import make_real_appraiser
    return make_real_appraiser(*args, **kwargs)


# ==========================================================================
# P3 real -- irreversibility / self-trap (mirrors G0.exp3, reduced)
# ==========================================================================
def run_p3_real(client, model, *, _make_appraiser=None, _scale=1.0):
    """Real-appraiser version of G0.exp3 at reduced size (lr_const=0.13).

    Phase 1 (inject_state=True): run(warmup+harm), capture s_dmg, A_secure
      (A just before harm) and A_damaged.
    Phase 2 from s_dmg with IDENTICAL heal stream:
      (a) inject_state=True  -> end_A_on
      (b) inject_state=False  (real "bias OFF" ablation) -> end_A_ablation

    ~382 calls at full size. `_scale` shrinks the streams for the self-test.

    Returns dict(A_secure, A_damaged, end_A_on, end_A_ablation, steps_on,
    steps_off, n_calls, hard_fails, fig).
    """
    from G0 import run, P
    from events_text import KEY_TO_TEXT

    if _make_appraiser is None:
        _make_appraiser = _default_make_appraiser

    LR = 0.13

    def _rep(base):
        return max(1, int(round(base * _scale)))

    warmup = ["nurture", "play"] * _rep(14)
    harm = ["harm", "scold"] * _rep(37)
    heal = ["nurture", "play"] * _rep(70)

    counter = _make_counter()

    # ---- phase 1: capture a secure creature into the damaged basin --------
    appr_on = _robust(
        _make_appraiser(client, model, KEY_TO_TEXT, text_pick="first",
                        inject_state=True),
        counter, label="P3 (state ON)")
    cap = run(warmup + harm, P, lr_const=LR, appraiser=appr_on)
    s_dmg = cap[-1].copy()
    A_secure = float(cap[len(warmup) - 1, 2])   # A just before harm
    A_damaged = float(s_dmg[2])

    # ---- phase 2a: identical kindness, state injected (ON) ----------------
    rec_on = run(heal, P, s0=s_dmg, lr_const=LR, appraiser=appr_on)

    # ---- phase 2b: identical kindness, ABLATION = inject_state=False ------
    appr_off = _robust(
        _make_appraiser(client, model, KEY_TO_TEXT, text_pick="first",
                        inject_state=False),
        counter, label="P3 (ablation)")
    rec_off = run(heal, P, s0=s_dmg, lr_const=LR, appraiser=appr_off)

    end_A_on = float(rec_on[-1, 2])
    end_A_ablation = float(rec_off[-1, 2])

    def steps_to(traj, thr=0.5):
        hit = np.where(traj[:, 2] >= thr)[0]
        return int(hit[0]) if len(hit) else None

    steps_on = steps_to(rec_on)
    steps_off = steps_to(rec_off)

    # ---- figure (like G0.exp3) --------------------------------------------
    fig_path = "g1_fig_p3.png"
    t_cap = np.arange(len(cap))
    t_rec = np.arange(len(cap), len(cap) + len(heal))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_cap, cap[:, 2], color="#aaaaaa", lw=1.8, label="life so far")
    ax.plot(t_rec, rec_on[:, 2], color="#ff2e88", lw=2.4,
            label="kindness, state ON (self-trapped)")
    ax.plot(t_rec, rec_off[:, 2], color="#00e5ff", lw=2.0, ls="--",
            label="kindness, state OFF ablation (recovers)")
    ax.axvspan(len(warmup), len(cap), color="#552222", alpha=0.5,
               label="harm burst")
    ax.axvspan(len(cap), len(cap) + len(heal), color="#225522", alpha=0.30,
               label="identical kindness")
    ax.axhline(0, color="#666", lw=0.7)
    ax.axhline(0.5, color="#888", lw=0.6, ls=":")
    ax.set_xlabel("interaction"); ax.set_ylabel("attachment A")
    ax.set_title("P3 real  irreversibility: secure A=%.2f -> harm -> A=%.2f -> "
                 "identical kindness  ON=%.2f  ablation=%.2f"
                 % (A_secure, A_damaged, end_A_on, end_A_ablation),
                 fontsize=10.5)
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)

    return dict(A_secure=A_secure, A_damaged=A_damaged, end_A_on=end_A_on,
                end_A_ablation=end_A_ablation, steps_on=steps_on,
                steps_off=steps_off, n_calls=counter["n_calls"],
                hard_fails=counter["hard_fails"], fig=fig_path)


# ==========================================================================
# P4 real -- timescale separation (mirrors G0.exp4)
# ==========================================================================
def run_p4_real(client, model, *, _make_appraiser=None, _scale=1.0):
    """Real-appraiser version of G0.exp4: one run over biased_childhood(300,
    0.5) with inject_state=True; compare autocorrelation times of mood v and
    trait A (same 1/e method as G0, re-implemented locally -- permitted).

    `_scale` shrinks the stream length for the self-test.

    Returns dict(mood_act, trait_act, ratio, n_calls, hard_fails, fig).
    """
    from G0 import run, P, biased_childhood
    from events_text import KEY_TO_TEXT

    if _make_appraiser is None:
        _make_appraiser = _default_make_appraiser

    n = max(10, int(round(300 * _scale)))
    stream = biased_childhood(n, 0.5)

    counter = _make_counter()
    appr = _robust(
        _make_appraiser(client, model, KEY_TO_TEXT, text_pick="first",
                        inject_state=True),
        counter, label="P4")
    tr = run(stream, P, appraiser=appr)
    v, A = tr[:, 0], tr[:, 2]

    def act(x):  # lag at which autocorr drops below 1/e (local re-impl, §5)
        x = x - x.mean()
        if np.allclose(x, 0):
            return 0
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac = ac / ac[0]
        below = np.where(ac < 1 / np.e)[0]
        return int(below[0]) if len(below) else len(x)

    mood_act, trait_act = act(v), act(A)
    ratio = trait_act / max(mood_act, 1)

    fig_path = "g1_fig_p4.png"
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)
    ax[0].plot(v, color="#00e5ff", lw=0.9)
    ax[0].set_ylabel("mood v (fast)")
    ax[0].set_title("P4 real  timescale separation:  mood tau~%d  vs  trait "
                    "tau~%d steps  (%.0fx)"
                    % (mood_act, trait_act, ratio), fontsize=11)
    ax[1].plot(A, color="#ff2e88", lw=2.0)
    ax[1].set_ylabel("trait A (slow)"); ax[1].set_xlabel("interaction")
    for a in ax:
        a.axhline(0, color="#555", lw=0.6)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)

    return dict(mood_act=mood_act, trait_act=trait_act, ratio=float(ratio),
                n_calls=counter["n_calls"], hard_fails=counter["hard_fails"],
                fig=fig_path)


# ==========================================================================
# P1 reduced -- multiple attractors (--full only)
# ==========================================================================
def run_p1_reduced(client, model, *, _make_appraiser=None, _scale=1.0,
                   _n_creatures=15):
    """Reduced real-appraiser P1: `_n_creatures` (default 15) creatures, each a
    biased_childhood(150, warmth~uniform(0.15,0.85)); real appraiser ON. Corner
    fraction = mean(|A|>0.6 & |R|>0.6). RNG seeded (default_rng(21)).

    Returns dict(corner_fraction, finals, n_calls, hard_fails, fig).
    """
    from G0 import run, P, biased_childhood
    from events_text import KEY_TO_TEXT

    if _make_appraiser is None:
        _make_appraiser = _default_make_appraiser

    rng = np.random.default_rng(21)
    n_events = max(10, int(round(150 * _scale)))

    counter = _make_counter()
    appr = _robust(
        _make_appraiser(client, model, KEY_TO_TEXT, text_pick="first",
                        inject_state=True),
        counter, label="P1")

    finals = []
    for _ in range(_n_creatures):
        w = float(rng.uniform(0.15, 0.85))
        stream = biased_childhood(n_events, w)
        finals.append(run(stream, P, appraiser=appr)[-1, 2:4])
    finals = np.array(finals)

    def cornered(d):
        return float(np.mean((np.abs(d[:, 0]) > 0.6) & (np.abs(d[:, 1]) > 0.6)))
    corner_fraction = cornered(finals)

    fig_path = "g1_fig_p1.png"
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(finals[:, 0], finals[:, 1], s=28, alpha=0.7, c="#00e5ff",
               edgecolors="none")
    for xx in (-1, 1):
        for yy in (-1, 1):
            ax.scatter([xx], [yy], marker="+", s=200, c="#ff2e88", lw=2)
    ax.axhline(0, color="#444", lw=0.7); ax.axvline(0, color="#444", lw=0.7)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel("A  (damaged  <->  secure)")
    ax.set_ylabel("R  (volatile  <->  serene)")
    ax.set_title("P1 real  %d creatures, random childhoods\ncorner-capture "
                 "%.0f%%" % (_n_creatures, 100 * corner_fraction), fontsize=11)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)

    return dict(corner_fraction=corner_fraction, finals=finals.tolist(),
                n_calls=counter["n_calls"], hard_fails=counter["hard_fails"],
                fig=fig_path)


# ==========================================================================
# P2 reduced -- path dependence (--full only)
# ==========================================================================
def run_p2_reduced(client, model, *, _make_appraiser=None, _scale=1.0,
                   _n_perms=20):
    """Reduced real-appraiser P2: a fixed multiset of 100 events (scaled from
    G0.exp2 proportions), `_n_perms` (default 20) random permutations; real
    appraiser ON. RNG seeded (default_rng(22)).

    Multiset (100): nurture 19, play 12, neutral 36, neglect 19, scold 9,
    harm 5.

    Returns dict(spread_std, early_corr, early_slope, finals, n_calls,
    hard_fails, fig).
    """
    from G0 import run, P
    from events_text import KEY_TO_TEXT

    if _make_appraiser is None:
        _make_appraiser = _default_make_appraiser

    base = (["nurture"] * 19 + ["play"] * 12 + ["neutral"] * 36 +
            ["neglect"] * 19 + ["scold"] * 9 + ["harm"] * 5)  # 100 events
    assert len(base) == 100, len(base)

    # optional shrink for the self-test: subsample the multiset deterministically
    if _scale != 1.0:
        k = max(8, int(round(len(base) * _scale)))
        base = base[:k]

    rng = np.random.default_rng(22)
    valence = {"nurture": 1, "play": .6, "neutral": 0, "neglect": -.5,
               "scold": -.7, "harm": -1}

    counter = _make_counter()
    appr = _robust(
        _make_appraiser(client, model, KEY_TO_TEXT, text_pick="first",
                        inject_state=True),
        counter, label="P2")

    finals, early_val = [], []
    for _ in range(_n_perms):
        order = list(rng.permutation(base))
        finals.append(float(run(order, P, appraiser=appr)[-1, 2]))
        q = max(1, len(order) // 4)
        early_val.append(float(np.mean([valence[e] for e in order[:q]])))
    finals = np.array(finals); early_val = np.array(early_val)

    spread_std = float(finals.std())
    if np.std(early_val) > 1e-9 and len(finals) >= 2:
        slope, intercept = np.polyfit(early_val, finals, 1)
        corr = float(np.corrcoef(early_val, finals)[0, 1])
    else:
        slope, intercept, corr = 0.0, float(finals.mean()), 0.0

    fig_path = "g1_fig_p2.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    ax[0].hist(finals, bins=min(24, max(4, _n_perms)), color="#00e5ff",
               alpha=0.85)
    ax[0].axvline(0, color="#ff2e88", lw=1.2)
    ax[0].set_xlabel("final attachment A"); ax[0].set_ylabel("creatures")
    ax[0].set_title("identical event multiset,\n%d random orderings" % _n_perms,
                    fontsize=11)
    ax[1].scatter(early_val, finals, s=24, alpha=0.6, c="#00e5ff",
                  edgecolors="none")
    xs = np.linspace(early_val.min(), early_val.max(), 20)
    ax[1].plot(xs, intercept + slope * xs, color="#ff2e88", lw=2)
    ax[1].set_xlabel("mean valence of FIRST quarter of life")
    ax[1].set_ylabel("final attachment A")
    ax[1].set_title("early events dominate\nr = %.2f  slope = %.2f"
                    % (corr, slope), fontsize=11)
    fig.suptitle("P2 real  path dependence: WHEN, not just how much", fontsize=13)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)

    return dict(spread_std=spread_std, early_corr=corr, early_slope=float(slope),
                finals=finals.tolist(), n_calls=counter["n_calls"],
                hard_fails=counter["hard_fails"], fig=fig_path)


# ==========================================================================
# Self-test (stub appraiser; drastically shortened streams via _scale)
# ==========================================================================
if __name__ == "__main__":
    import os

    print("=== experiments_real self-test ===")

    # A stub `make_real_appraiser`: returns a plain appraiser(ev, s, age, p,
    # bias_on) -> (pv, pa, nov) with the SAME signature the seam produces, so
    # run() drives it exactly like the real thing -- but no LLM. Dynamics come
    # from G0.run (we never re-implement them).
    _stub_rng = np.random.default_rng(1)
    # rough per-key perceived valence so the trajectories move plausibly.
    _PV = {"nurture": 0.9, "play": 0.6, "neutral": 0.0, "neglect": -0.5,
           "scold": -0.7, "harm": -1.0}
    _NOV = {"nurture": 0.1, "play": 0.7, "neutral": 0.1, "neglect": 0.0,
            "scold": 0.1, "harm": 0.3}

    def stub_make_appraiser(client, model, key_to_text, name="the creature",
                            text_pick="first", inject_state=True):
        def appr(ev, s, age, p, bias_on):
            A = s[2]
            pv = _PV[ev]
            # crude state bias when injected: damaged (A<0) reads ambiguity down
            if inject_state:
                amb = 1.0 - abs(pv)
                pv = pv - 0.6 * max(0.0, -A) * amb
            pv = float(np.clip(pv + _stub_rng.normal(0, 0.02), -1, 1))
            pa = float(np.clip(0.3 * abs(_PV[ev]) + 0.2, 0, 1))
            return pv, pa, _NOV[ev]
        return appr

    # --- P3 (shortened streams via _scale) ---------------------------------
    # warmup*2, harm*4, heal*6 target: scale ~ {2/14, 4/37, 6/70}; use a small
    # uniform scale that keeps all three streams tiny.
    p3 = run_p3_real(None, "stub", _make_appraiser=stub_make_appraiser,
                     _scale=0.12)
    req3 = {"A_secure", "A_damaged", "end_A_on", "end_A_ablation", "steps_on",
            "steps_off", "n_calls", "hard_fails", "fig"}
    assert req3 <= set(p3), req3 - set(p3)
    assert os.path.exists(p3["fig"]) and p3["fig"] == "g1_fig_p3.png"
    assert p3["hard_fails"] == 0
    print(f"  P3 OK: A_secure={p3['A_secure']:+.2f} A_damaged={p3['A_damaged']:+.2f} "
          f"end_on={p3['end_A_on']:+.2f} end_ablation={p3['end_A_ablation']:+.2f} "
          f"n_calls={p3['n_calls']} fig={p3['fig']}")

    # --- P4 ----------------------------------------------------------------
    p4 = run_p4_real(None, "stub", _make_appraiser=stub_make_appraiser,
                     _scale=0.1)
    req4 = {"mood_act", "trait_act", "ratio", "n_calls", "hard_fails", "fig"}
    assert req4 <= set(p4), req4 - set(p4)
    assert os.path.exists(p4["fig"]) and p4["fig"] == "g1_fig_p4.png"
    print(f"  P4 OK: mood_act={p4['mood_act']} trait_act={p4['trait_act']} "
          f"ratio={p4['ratio']:.1f} n_calls={p4['n_calls']} fig={p4['fig']}")

    # --- P1 reduced (few creatures) ----------------------------------------
    p1 = run_p1_reduced(None, "stub", _make_appraiser=stub_make_appraiser,
                        _scale=0.1, _n_creatures=3)
    req1 = {"corner_fraction", "finals", "n_calls", "hard_fails", "fig"}
    assert req1 <= set(p1), req1 - set(p1)
    assert os.path.exists(p1["fig"]) and p1["fig"] == "g1_fig_p1.png"
    assert len(p1["finals"]) == 3
    print(f"  P1 OK: corner_fraction={p1['corner_fraction']:.2f} "
          f"finals={len(p1['finals'])} n_calls={p1['n_calls']} fig={p1['fig']}")

    # --- P2 reduced (few perms) --------------------------------------------
    p2 = run_p2_reduced(None, "stub", _make_appraiser=stub_make_appraiser,
                        _scale=0.15, _n_perms=4)
    req2 = {"spread_std", "early_corr", "early_slope", "finals", "n_calls",
            "hard_fails", "fig"}
    assert req2 <= set(p2), req2 - set(p2)
    assert os.path.exists(p2["fig"]) and p2["fig"] == "g1_fig_p2.png"
    assert len(p2["finals"]) == 4
    print(f"  P2 OK: spread_std={p2['spread_std']:.3f} "
          f"early_corr={p2['early_corr']:+.2f} early_slope={p2['early_slope']:+.2f} "
          f"finals={len(p2['finals'])} n_calls={p2['n_calls']} fig={p2['fig']}")

    # --- hard-fail path: stub whose appraiser raises AppraisalError --------
    try:
        from parse import AppraisalError as _AE
    except Exception:
        class _AE(Exception):
            pass

    def stub_make_failing(client, model, key_to_text, name="the creature",
                          text_pick="first", inject_state=True):
        def appr(ev, s, age, p, bias_on):
            raise _AE("boom")
        return appr

    p3f = run_p3_real(None, "stub", _make_appraiser=stub_make_failing,
                      _scale=0.1)
    # every logical event falls back to neutral -> hard_fails == n_calls
    assert p3f["hard_fails"] == p3f["n_calls"] and p3f["n_calls"] > 0
    print(f"  hard-fail path OK: every call fell back, "
          f"hard_fails={p3f['hard_fails']} == n_calls={p3f['n_calls']}")

    # --- OPTIONAL: 1 real call through the seam if the stack imports --------
    real_ok = False
    try:
        from appraiser import appraise_llm  # noqa: F401
        from ollama_client import make_client
        from seam import make_real_appraiser  # noqa: F401
        real_ok = True
    except Exception as e:
        print(f"  [skip] real seam call: appraiser stack unavailable ({e})")

    if real_ok:
        try:
            from G0 import run, P
            from events_text import KEY_TO_TEXT
            client = make_client("http://localhost:11434")
            # Sanity-check the real seam WITHOUT run_p4_real's 10-event floor:
            # drive run() over a 2-event stream through the real appraiser, so
            # exactly ~2 real LLM calls (stays inside the C-file call budget).
            counter = _make_counter()
            appr = _robust(
                make_real_appraiser(client, "qwen3.5:9b", KEY_TO_TEXT,
                                    text_pick="first", inject_state=True),
                counter)
            tr = run(["neutral", "nurture"], P, appraiser=appr)
            assert tr.shape == (2, 5)
            print(f"  [real] seam call OK: {counter['n_calls']} real calls, "
                  f"hard_fails={counter['hard_fails']}, final A={tr[-1,2]:+.2f} "
                  f"(2-event stream, not a result)")
        except Exception as e:
            print(f"  [skip] real seam call failed at runtime ({e})")

    print("=== self-test PASSED ===")
