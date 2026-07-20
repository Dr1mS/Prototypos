"""
G0 harness -- personality attractor dynamics, LLM mocked, no UI.

Goal: empirically prove 4 properties of the trait-dynamics rule BEFORE
building anything on top of it. If these curves don't come out, nothing
downstream matters.

  P1  multiple attractors reachable      (not a single resting point)
  P2  path dependence (order matters)     (same event multiset -> diff outcome)
  P3  hysteresis / scarring               (kindness can't fully undo harm)
  P4  timescale separation                (mood flickers, traits stay stable)

State (two timescales):
  mood   : v (valence), r (arousal)  -- fast, leaky, drives traits
  traits : A (attachment: secure +1 / damaged -1)  -- slow, BISTABLE
           R (regulation: serene +1 / volatile -1)  -- slow, BISTABLE
           O (openness)                              -- slow, SIMPLE integrator

Core thesis under test: the personality state biases PERCEPTION (the mocked
appraiser), not just expression. That single feedback is what turns a plain
relaxation into a multi-attractor, path-dependent, self-scarring system.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- parameters
P = dict(
    # mood (fast)
    kv=0.55, kr=0.55,            # mood relaxation speed toward perceived target
    # trait double-well  dx = (a*x - b*x^3 + drive) * lr
    # a=b=0.85 keeps wells at +-1 but widens the drive-vs-barrier margin so a
    # trauma burst can cross into the damaged basin (not just a childhood).
    aA=0.85, bA=0.85, wA=0.42,   # attachment
    aR=0.85, bR=0.85, wR=0.42,   # regulation
    # openness simple integrator
    wO=0.25, leakO=0.15, Orest=0.0,
    # plasticity annealing  lr(age) = lr0 * exp(-age/tau)
    lr0=0.16, tau=80.0,
    # perception bias (the thesis): damaged creature reads ambiguity as threat
    k_bias=0.9,
    # noise
    trait_noise=0.010,
)

# event -> RAW appraisal primitives (before perception bias)
#   care  in [-1,1]  nurture vs neglect/harm
#   thr   in [0,1]   threat
#   nov   in [0,1]   novelty
#   aut   in [-1,1]  autonomy respected vs controlled
EVENTS = {
    "nurture": dict(care=+1.0, thr=0.0, nov=0.1, aut=0.0),
    "play":    dict(care=+0.6, thr=0.0, nov=0.7, aut=+0.3),
    "neutral": dict(care=+0.0, thr=0.0, nov=0.1, aut=0.0),   # ambiguous
    "neglect": dict(care=-0.5, thr=0.0, nov=0.0, aut=0.0),   # ambiguous absence
    "scold":   dict(care=-0.7, thr=0.4, nov=0.1, aut=-0.6),
    "harm":    dict(care=-1.0, thr=0.9, nov=0.3, aut=-0.8),
}


def appraise(ev, A, k_bias):
    """Perception biased by current attachment A. THE core loop."""
    e = EVENTS[ev]
    care, thr, nov, aut = e["care"], e["thr"], e["nov"], e["aut"]
    bias = k_bias * max(0.0, -A)          # only fires when damaged (A<0)
    amb = 1.0 - abs(care)                 # ambiguity, max at care=0
    pcare = care - bias * amb             # neutral/absence read as negative
    if care > 0:
        pcare -= bias * 0.5 * care        # kindness discounted when damaged
    pv = pcare - 0.5 * thr + 0.3 * aut    # perceived valence
    pa = 0.7 * thr + 0.5 * nov            # perceived arousal
    return np.clip(pv, -1, 1), np.clip(pa, 0, 1), nov


def mock_appraiser(ev, s, age, p, bias_on):
    """Default appraiser: the existing mock, unchanged (G1 seam, §1.5.c)."""
    A = s[2]
    k_bias = p["k_bias"] if bias_on else 0.0
    return appraise(ev, A, k_bias)


def step(s, ev, age, p, bias_on=True, lr_const=None, linear=False,
         appraiser=None):
    v, r, A, R, O = s
    if appraiser is None:
        appraiser = mock_appraiser
    pv, pa, nov = appraiser(ev, s, age, p, bias_on)   # <- the only seam change
    # mood (fast relaxation toward perceived target)
    v += p["kv"] * (pv - v)
    r += p["kr"] * (pa - r)
    # plasticity  (lr_const overrides annealing to isolate other mechanisms)
    lr = lr_const if lr_const is not None else p["lr0"] * np.exp(-age / p["tau"])
    if linear:  # single-attractor control: leaky linear integrator, no well
        leak = 0.15
        dA = (p["wA"] * v - leak * A) * lr
        dR = (p["wR"] * (-r + 0.3 * v) - leak * R) * lr
        dO = (p["wO"] * nov - p["leakO"] * (O - p["Orest"])) * lr
        n = p["trait_noise"]
        A = np.clip(A + dA + RNG.normal(0, n), -1.2, 1.2)
        R = np.clip(R + dR + RNG.normal(0, n), -1.2, 1.2)
        O = np.clip(O + dO + RNG.normal(0, n * 0.5), -1.2, 1.2)
        return np.array([v, r, A, R, O])
    # traits (slow). bistable A,R driven by sustained mood; O simple.
    dA = (p["aA"] * A - p["bA"] * A**3 + p["wA"] * v) * lr
    dR = (p["aR"] * R - p["bR"] * R**3 + p["wR"] * (-r + 0.3 * v)) * lr
    dO = (p["wO"] * nov - p["leakO"] * (O - p["Orest"])) * lr
    n = p["trait_noise"]
    A = np.clip(A + dA + RNG.normal(0, n), -1.2, 1.2)
    R = np.clip(R + dR + RNG.normal(0, n), -1.2, 1.2)
    O = np.clip(O + dO + RNG.normal(0, n * 0.5), -1.2, 1.2)
    return np.array([v, r, A, R, O])


def run(stream, p, s0=None, bias_on=True, lr_const=None, linear=False,
        appraiser=None):
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) if s0 is None else np.array(s0, float)
    traj = np.empty((len(stream), 5))
    for t, ev in enumerate(stream):
        s = step(s, ev, t, p, bias_on, lr_const, linear, appraiser)
        traj[t] = s
    return traj  # columns: v, r, A, R, O


# ---------------------------------------------------------- stream builders
def biased_childhood(n=300, warmth=0.5):
    """Random life with a per-creature warmth bias in [0,1]."""
    pos = ["nurture", "play", "neutral"]
    neg = ["neglect", "scold", "harm", "neutral"]
    out = []
    for _ in range(n):
        out.append(RNG.choice(pos) if RNG.random() < warmth else RNG.choice(neg))
    return out


# =============================================================== EXPERIMENTS
def exp1_multiattractor(fname="fig1_attractors.png"):
    """P1: many creatures, random childhoods -> final (A,R) cluster in basins,
    not a blob at centre. Ablation: perception bias OFF collapses the wells."""
    N = 250
    finals_nl, finals_lin = [], []
    for _ in range(N):
        w = RNG.uniform(0.15, 0.85)
        stream = biased_childhood(300, w)
        finals_nl.append(run(stream, P, bias_on=True)[-1, 2:4])
        finals_lin.append(run(stream, P, bias_on=True, linear=True)[-1, 2:4])
    on = np.array(finals_nl); off = np.array(finals_lin)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    for a, dat, title in [(ax[0], on, "double-well (this design)"),
                          (ax[1], off, "linear integrator (control)")]:
        a.scatter(dat[:, 0], dat[:, 1], s=16, alpha=0.6, c="#00e5ff",
                  edgecolors="none")
        for xx in (-1, 1):
            for yy in (-1, 1):
                a.scatter([xx], [yy], marker="+", s=200, c="#ff2e88", lw=2)
        a.axhline(0, color="#444", lw=0.7); a.axvline(0, color="#444", lw=0.7)
        a.set_xlim(-1.25, 1.25); a.set_ylim(-1.25, 1.25)
        a.set_xlabel("A  (damaged  <->  secure)")
        a.set_ylabel("R  (volatile  <->  serene)")
        a.set_title(title, fontsize=11)
    fig.suptitle("P1  multiple attractors: 250 creatures, random childhoods",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)

    # metric: fraction landing near a corner (|A|>0.6 and |R|>0.6)
    def cornered(d):
        return np.mean((np.abs(d[:, 0]) > 0.6) & (np.abs(d[:, 1]) > 0.6))
    return dict(corner_nl=cornered(on), corner_lin=cornered(off))


def exp2_pathdependence(fname="fig2_pathdep.png"):
    """P2: ONE fixed event multiset, many random orderings -> spread of final A.
    Plus: mean valence of the FIRST QUARTER predicts final A (early dominance)."""
    base = (["nurture"] * 22 + ["play"] * 14 + ["neutral"] * 40 +
            ["neglect"] * 22 + ["scold"] * 10 + ["harm"] * 6)  # fixed multiset
    K = 120
    finals, early_val = [], []
    valence = {"nurture": 1, "play": .6, "neutral": 0, "neglect": -.5,
               "scold": -.7, "harm": -1}
    for _ in range(K):
        order = list(RNG.permutation(base))
        finals.append(run(order, P, bias_on=True)[-1, 2])
        q = len(order) // 4
        early_val.append(np.mean([valence[e] for e in order[:q]]))
    finals = np.array(finals); early_val = np.array(early_val)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    ax[0].hist(finals, bins=24, color="#00e5ff", alpha=0.85)
    ax[0].axvline(0, color="#ff2e88", lw=1.2)
    ax[0].set_xlabel("final attachment A"); ax[0].set_ylabel("creatures")
    ax[0].set_title("identical event multiset,\n%d random orderings" % K,
                    fontsize=11)
    ax[1].scatter(early_val, finals, s=20, alpha=0.6, c="#00e5ff",
                  edgecolors="none")
    b, a0 = np.polyfit(early_val, finals, 1)
    xs = np.linspace(early_val.min(), early_val.max(), 20)
    ax[1].plot(xs, a0 + b * xs, color="#ff2e88", lw=2)
    ax[1].set_xlabel("mean valence of FIRST quarter of life")
    ax[1].set_ylabel("final attachment A")
    r = np.corrcoef(early_val, finals)[0, 1]
    ax[1].set_title("early events dominate\nr = %.2f  slope = %.2f" % (r, b),
                    fontsize=11)
    fig.suptitle("P2  path dependence: WHEN, not just how much", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)
    return dict(spread_std=float(finals.std()), early_corr=float(r),
                early_slope=float(b))


def exp3_hysteresis(fname="fig3_scar.png"):
    """P3: irreversibility as a SELF-TRAP. Plasticity constant (no annealing
    confound). Phase 1: a secure creature is captured into the damaged basin by
    a harm burst. Phase 2: from that same damaged state, apply identical
    kindness with perception bias ON vs OFF. ON is self-trapping (kindness is
    discounted, can't climb the barrier); OFF climbs back out. The scar comes
    from the SAME perception loop that builds the attractors -- one mechanism."""
    LR = 0.13
    # ---- phase 1: capture a secure creature into the damaged basin
    warmup = ["nurture", "play"] * 14
    harm = ["harm", "scold"] * 37              # long enough to capture the basin
    cap = run(warmup + harm, P, bias_on=True, lr_const=LR)
    s_dmg = cap[-1].copy()                      # damaged state, shared start
    a_secure = float(cap[len(warmup) - 1, 2])   # A just before harm
    a_dmg = float(s_dmg[2])

    # ---- phase 2: identical kindness from the identical damaged state
    heal = ["nurture", "play"] * 70
    rec_on = run(heal, P, s0=s_dmg, bias_on=True, lr_const=LR)
    rec_off = run(heal, P, s0=s_dmg, bias_on=False, lr_const=LR)

    def steps_to(traj, thr=0.5):
        hit = np.where(traj[:, 2] >= thr)[0]
        return int(hit[0]) if len(hit) else None

    k_on, k_off = steps_to(rec_on), steps_to(rec_off)

    # assemble full display trajectory (capture then ON recovery)
    t_cap = np.arange(len(cap))
    t_rec = np.arange(len(cap), len(cap) + len(heal))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_cap, cap[:, 2], color="#aaaaaa", lw=1.8, label="life so far")
    ax.plot(t_rec, rec_on[:, 2], color="#ff2e88", lw=2.4,
            label="kindness, bias ON (self-trapped)")
    ax.plot(t_rec, rec_off[:, 2], color="#00e5ff", lw=2.0, ls="--",
            label="kindness, bias OFF (recovers)")
    ax.axvspan(len(warmup), len(cap), color="#552222", alpha=0.5,
               label="harm burst")
    ax.axvspan(len(cap), len(cap) + len(heal), color="#225522", alpha=0.30,
               label="identical kindness")
    ax.axhline(0, color="#666", lw=0.7)
    ax.axhline(0.5, color="#888", lw=0.6, ls=":")
    end_on, end_off = float(rec_on[-1, 2]), float(rec_off[-1, 2])
    ax.set_xlabel("interaction"); ax.set_ylabel("attachment A")
    ax.set_title("P3  irreversibility: secure A=%.2f -> harm -> A=%.2f -> after "
                 "identical kindness  ON=%.2f  OFF=%.2f"
                 % (a_secure, a_dmg, end_on, end_off), fontsize=10.5)
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)
    return dict(secure=a_secure, damaged=a_dmg, end_on=end_on, end_off=end_off,
                steps_on=k_on, steps_off=k_off)


def exp4_timescales(fname="fig4_timescales.png"):
    """P4: same stream, plot mood v vs trait A. Compare autocorrelation time:
    mood short (flickers), trait long (stable character)."""
    stream = biased_childhood(300, 0.5)
    tr = run(stream, P, bias_on=True)
    v, A = tr[:, 0], tr[:, 2]

    def act(x):  # lag at which autocorr drops below 1/e
        x = x - x.mean()
        if np.allclose(x, 0):
            return 0
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac = ac / ac[0]
        below = np.where(ac < 1 / np.e)[0]
        return int(below[0]) if len(below) else len(x)

    tv, ta = act(v), act(A)
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)
    ax[0].plot(v, color="#00e5ff", lw=0.9)
    ax[0].set_ylabel("mood v (fast)")
    ax[0].set_title("P4  timescale separation:  mood tau~%d  vs  trait tau~%d "
                    "steps  (%.0fx)" % (tv, ta, ta / max(tv, 1)), fontsize=11)
    ax[1].plot(A, color="#ff2e88", lw=2.0)
    ax[1].set_ylabel("trait A (slow)"); ax[1].set_xlabel("interaction")
    for a in ax:
        a.axhline(0, color="#555", lw=0.6)
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)
    return dict(mood_act=tv, trait_act=ta, ratio=ta / max(tv, 1))


if __name__ == "__main__":
    print("=== G0 harness ===")
    r1 = exp1_multiattractor()
    print("P1 multi-attractor : corner-capture double-well=%.0f%%  linear=%.0f%%"
          % (100 * r1["corner_nl"], 100 * r1["corner_lin"]))
    r2 = exp2_pathdependence()
    print("P2 path dependence : final-A std over orderings=%.2f  "
          "early-quarter corr=%.2f (slope %.2f)"
          % (r2["spread_std"], r2["early_corr"], r2["early_slope"]))
    r3 = exp3_hysteresis()
    print("P3 irreversibility : secure A=%.2f -> harm -> A=%.2f ; identical "
          "kindness -> ON=%.2f (steps=%s)  OFF=%.2f (steps=%s)"
          % (r3["secure"], r3["damaged"], r3["end_on"], r3["steps_on"],
             r3["end_off"], r3["steps_off"]))
    r4 = exp4_timescales()
    print("P4 timescales      : mood ACT=%d  trait ACT=%d  ratio=%.0fx"
          % (r4["mood_act"], r4["trait_act"], r4["ratio"]))
    print("figs: fig1_attractors fig2_pathdep fig3_scar fig4_timescales")