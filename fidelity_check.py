"""fidelity_check.py -- G1.5 §5 field-fidelity spot-check (supervisor-owned).

Frozen design (prereg_g15.md, PRED-5):
  10 creatures x 250 events, warmth = linspace(0.2, 0.8, 10), streams from
  experiments_field.make_stream with rng seed 51. Each stream is run twice:
  real path (appraise_llm through the seam, text_pick="first", inject_state
  =True) and field path (make_field_appraiser, noise ON, seed 31, ONE shared
  appraiser instance across the 10 runs). G0.RNG is reseeded identically
  (default_rng(1000+i)) before each member of a pair so the harness trait
  noise is pair-matched and the only difference is the appraiser.

  PASS iff (a) |mean(final A, real) - mean(final A, field)| < 0.30, and
  (b) |A_real - A_field| < 0.50 for >= 7/10 matched pairs.

~2 500 real LLM calls (~2h at measured ~3.3 s/call). Run AFTER subagent D has
finished (single Ollama client at a time).
"""
import json
import time

import numpy as np

import G0
from events_text import KEY_TO_TEXT
from experiments_field import make_stream
from experiments_real import _robust, _make_counter
from field_appraiser import make_field_appraiser
from ollama_client import make_client
from seam import make_real_appraiser

MODEL = "qwen3.5:9b"
N_CREATURES = 10
STREAM_LEN = 250
WARMTHS = np.linspace(0.2, 0.8, N_CREATURES)
STREAM_SEED = 51
FIELD_SEED = 31
PAIR_SEED_BASE = 1000

MEAN_TOL = 0.30      # PRED-5 (a)
PAIR_TOL = 0.50      # PRED-5 (b)
PAIR_MIN = 7         # PRED-5 (b): >= 7/10 pairs


def main():
    rng = np.random.default_rng(STREAM_SEED)
    streams = [make_stream(rng, STREAM_LEN, w) for w in WARMTHS]

    # ---- field replays first (seconds, no LLM) ----------------------------
    field_appr = make_field_appraiser("field_A.json", noise=True,
                                      seed=FIELD_SEED)
    finals_field = []
    for i, stream in enumerate(streams):
        G0.RNG = np.random.default_rng(PAIR_SEED_BASE + i)
        traj = G0.run(stream, G0.P, appraiser=field_appr)
        finals_field.append(traj[-1])
    print("field replays done (10/10)")

    # ---- real runs (~2 500 LLM calls) -------------------------------------
    client = make_client()
    counter = _make_counter()
    finals_real = []
    t0 = time.time()
    for i, stream in enumerate(streams):
        G0.RNG = np.random.default_rng(PAIR_SEED_BASE + i)
        appr = _robust(
            make_real_appraiser(client, MODEL, KEY_TO_TEXT,
                                text_pick="first", inject_state=True),
            counter, label=f"real-{i}", every=50)
        traj = G0.run(stream, G0.P, appraiser=appr)
        finals_real.append(traj[-1])
        el = (time.time() - t0) / 60
        print(f"  real run {i + 1}/{N_CREATURES} done "
              f"(warmth {WARMTHS[i]:.2f}, final A {traj[-1][2]:+.2f}, "
              f"{el:.0f} min elapsed)")

    finals_real = np.array(finals_real)
    finals_field = np.array(finals_field)
    a_real, a_field = finals_real[:, 2], finals_field[:, 2]

    # ---- PRED-5 evaluation -------------------------------------------------
    mean_diff = float(abs(a_real.mean() - a_field.mean()))
    pair_diffs = np.abs(a_real - a_field)
    pairs_ok = int((pair_diffs < PAIR_TOL).sum())
    ok_a = mean_diff < MEAN_TOL
    ok_b = pairs_ok >= PAIR_MIN
    verdict = "PASS" if (ok_a and ok_b) else "FAIL"

    print("\n" + "=" * 60)
    print(f"  mean final A: real {a_real.mean():+.3f}  "
          f"field {a_field.mean():+.3f}  |diff| {mean_diff:.3f} "
          f"({'OK' if ok_a else 'FAIL'} < {MEAN_TOL})")
    print(f"  matched pairs |dA| < {PAIR_TOL}: {pairs_ok}/10 "
          f"({'OK' if ok_b else 'FAIL'} >= {PAIR_MIN})")
    print(f"  per-pair |dA|: {np.round(pair_diffs, 2).tolist()}")
    print(f"  hard fails on real path: {counter['hard_fails']} "
          f"/ {counter['n_calls']} calls")
    print(f"PRED-5 field-fidelity: {verdict}")
    print("=" * 60)

    # ---- figure + results --------------------------------------------------
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([-1.3, 1.3], [-1.3, 1.3], color="#666", lw=0.8, ls="--")
    ax.scatter(a_field, a_real, s=60, c="#00e5ff", edgecolors="none")
    for i in range(N_CREATURES):
        ax.annotate(f"{WARMTHS[i]:.2f}", (a_field[i], a_real[i]),
                    fontsize=7, color="#aaa", xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("final A -- field-driven")
    ax.set_ylabel("final A -- real LLM")
    ax.set_title(f"G1.5 fidelity spot-check: {verdict} "
                 f"(mean diff {mean_diff:.2f}, pairs {pairs_ok}/10)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("g15_fig_fidelity.png", dpi=120)
    plt.close(fig)

    out = dict(
        verdict=verdict, mean_diff=mean_diff, pairs_ok=pairs_ok,
        pair_diffs=pair_diffs.tolist(),
        warmths=WARMTHS.tolist(),
        finals_real=finals_real.tolist(), finals_field=finals_field.tolist(),
        n_calls=counter["n_calls"], hard_fails=counter["hard_fails"],
        elapsed_min=round((time.time() - t0) / 60, 1),
        params=dict(n=N_CREATURES, length=STREAM_LEN, stream_seed=STREAM_SEED,
                    field_seed=FIELD_SEED, pair_seed_base=PAIR_SEED_BASE,
                    model=MODEL),
    )
    with open("g15_fidelity_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("written: g15_fidelity_results.json, g15_fig_fidelity.png")


if __name__ == "__main__":
    main()
