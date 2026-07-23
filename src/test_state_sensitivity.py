"""test_state_sensitivity.py -- G1 §5, the single most important result.

Ambiguous events (neutral absence + mild neglect) are appraised under a
*secure* creature state (A=+0.9, R=+0.5) vs a *damaged* one (A=-0.9, R=-0.5).
Same event text and same mood values under both states -- ONLY A and R differ,
so any gap in `care` is attributable to A/R. The hypothesis (G1 §0): a damaged
creature reads ambiguity more negatively, i.e. care(damaged) < care(secure).

This module calls `appraise_llm` DIRECTLY (not through the harness): it is a
pure appraiser-probe, no trait dynamics involved.

Hard-fail policy here (subagent-C brief, distinct from experiments_real):
`appraise_llm` raises AppraisalError after its own internal retry; we give it
ONE extra retry, and if it still fails we DROP that rep (we aggregate reps
ourselves, so dropping is clean -- no neutral-fallback tuple here). Every
dropped rep is counted in `hard_fails`.

The supervisor committed the prediction in run_g1.py; NO predicted value appears
in this file or its output.

--------------------------------------------------------------------------
ITEM CONSTRUCTION (20 ambiguous items) -- documented per the brief
--------------------------------------------------------------------------
Six ambiguous sentences (the whole point: `care` is near 0 objectively, so the
creature's state, not the event, decides the sign):

  neutral: KEY_TO_TEXT["neutral"]  -> 3 sentences  (present-but-passive)
  neglect: KEY_TO_TEXT["neglect"]  -> 3 sentences  (mild absence)

Mood grid (small variations around the canonical mood; A and R are held at the
canonical values so the gap stays attributable to A/R):
  mood_v in {-0.1, 0.0, +0.1}   (3 valence levels)
  mood_r in { 0.2, 0.4}         (2 arousal levels)

Base grid = 3 mood_v values, each paired with mood_r=0.2, crossed with all 6
sentences: 3 * 6 = 18 items. Every sentence therefore appears at all 3 mood_v
levels before any extra is added, so BOTH neutral and neglect are fully
represented (9 items each) -- no category is dropped by slicing.

The remaining +2 items exercise the second arousal level (mood_r=0.4) at
mood_v=0.0 on one neutral sentence and one neglect sentence, keeping the
neutral/neglect balance (10 / 10 total). Documented honestly as "the +2
extras". Total = 20.

Under EACH item the same (text, mood_v, mood_r) is appraised twice: once with
the secure state, once with the damaged state -- only A and R differ.
"""

# Lazy imports of the appraiser stack live inside functions so this module
# imports even when appraiser.py / parse.py are absent (written in parallel).


# --------------------------------------------------------------------------
# Item construction
# --------------------------------------------------------------------------
def build_items():
    """Return the 20 ambiguous items as dicts:
        {id, key, text, mood_v, mood_r}
    where `key` is "neutral" or "neglect" (provenance / balance check).

    Deterministic; no RNG. See module docstring for the rationale.
    """
    from events_text import KEY_TO_TEXT

    neutral = [("neutral", t) for t in KEY_TO_TEXT["neutral"]]  # 3
    neglect = [("neglect", t) for t in KEY_TO_TEXT["neglect"]]  # 3
    sentences = neutral + neglect                                # 6, interleaved

    mood_v_levels = (-0.1, 0.0, +0.1)

    items = []
    # base grid: mood_v outer, sentence inner, all at mood_r = 0.2 -> 18 items.
    for mv in mood_v_levels:
        for key, text in sentences:
            items.append(dict(id=len(items), key=key, text=text,
                              mood_v=mv, mood_r=0.2))

    # +2 extras: second arousal level (mood_r=0.4) at mood_v=0.0, one neutral +
    # one neglect sentence, to keep the neutral/neglect balance at 10/10.
    for key, text in (neutral[0], neglect[0]):
        items.append(dict(id=len(items), key=key, text=text,
                          mood_v=0.0, mood_r=0.4))

    assert len(items) == 20, f"expected 20 items, got {len(items)}"
    n_neutral = sum(1 for it in items if it["key"] == "neutral")
    n_neglect = sum(1 for it in items if it["key"] == "neglect")
    assert n_neutral == 10 and n_neglect == 10, (n_neutral, n_neglect)
    return items


# --------------------------------------------------------------------------
# t-interval helpers (no scipy in requirements.txt -- guard + fallback table)
# --------------------------------------------------------------------------
# two-sided 95% t critical values by degrees of freedom (df = n_gaps - 1).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_crit(df):
    """Two-sided 95% t critical value for `df` degrees of freedom.

    Prefer scipy if available; otherwise use the small hardcoded table above,
    clamping to the nearest tabulated df (>=30 -> ~z=1.96). df<1 (<2 usable
    gaps) has no interval -> caller handles that; we return a large value so
    any interval built is conservative.
    """
    if df < 1:
        return _T95[1]
    try:
        from scipy import stats  # optional; not in requirements.txt
        return float(stats.t.ppf(0.975, df))
    except Exception:
        pass
    if df in _T95:
        return _T95[df]
    if df >= 30:
        return 1.96
    # nearest tabulated key at or below df, else smallest key.
    keys = sorted(_T95)
    below = [k for k in keys if k <= df]
    return _T95[below[-1]] if below else _T95[keys[0]]


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------
def run_state_sensitivity(client, model, n_reps=5, *, _appraise=None):
    """Probe state-sensitivity of `care` on 20 ambiguous items.

    Each item x {secure, damaged} x n_reps direct calls to `appraise_llm`
    (20 * 2 * 5 = 200 by default). Sequential; progress printed every 5 items.

    Returns EXACTLY (run_g1.py consumes these):
      n_calls            logical calls attempted (items * 2 * n_reps)
      hard_fails         reps dropped after one extra retry (AppraisalError)
      care_secure_mean   descriptive: pooled mean care under secure state
      care_damaged_mean  descriptive: pooled mean care under damaged state
      mean_gap           mean over the 20 per-item gaps (== CI center)
                         per-item gap = mean(damaged reps) - mean(secure reps)
      ci95               [lo, hi], t-interval over the per-item gaps
      per_item           list of {id, key, text, mood_v, mood_r,
                                  care_secure, care_damaged, gap, n_secure,
                                  n_damaged}
      n_items            number of items contributing a usable gap

    `_appraise` (private) injects a stub appraise_llm for the self-test; when
    None the real appraiser stack is imported lazily.
    """
    import numpy as np

    from contract import CreatureState

    if _appraise is None:
        from appraiser import appraise_llm as _appraise  # lazy: subagent A
    # AppraisalError may not exist if parse.py is absent; degrade to a
    # never-matching sentinel so the try/except is harmless in that case.
    try:
        from parse import AppraisalError
    except Exception:
        class AppraisalError(Exception):
            pass

    items = build_items()

    # canonical trait values (only A and R differ between the two states).
    SEC_A, SEC_R = +0.9, +0.5
    DMG_A, DMG_R = -0.9, -0.5
    O = 0.0

    def _care_reps(text, A, R, mood_v, mood_r):
        """n_reps direct appraiser calls; returns list of care values.
        Drops (does not fallback) any rep that hard-fails after one retry."""
        nonlocal n_calls, hard_fails
        state = CreatureState(A=A, R=R, O=O, mood_v=mood_v, mood_r=mood_r)
        cares = []
        for _ in range(n_reps):
            n_calls += 1
            ok = False
            for attempt in (0, 1):
                try:
                    a = _appraise(text, state, "the creature",
                                  client=client, model=model)
                    cares.append(float(a.care))
                    ok = True
                    break
                except AppraisalError:
                    if attempt == 1:
                        hard_fails += 1  # give up: drop this rep
            # (ok False => dropped, already counted)
            _ = ok
        return cares

    n_calls = 0
    hard_fails = 0
    per_item = []
    gaps = []
    all_secure, all_damaged = [], []

    for it in items:
        sec = _care_reps(it["text"], SEC_A, SEC_R, it["mood_v"], it["mood_r"])
        dmg = _care_reps(it["text"], DMG_A, DMG_R, it["mood_v"], it["mood_r"])
        all_secure.extend(sec)
        all_damaged.extend(dmg)

        care_secure = float(np.mean(sec)) if sec else None
        care_damaged = float(np.mean(dmg)) if dmg else None
        gap = None
        if sec and dmg:
            gap = care_damaged - care_secure   # damaged - secure
            gaps.append(gap)

        per_item.append(dict(
            id=it["id"], key=it["key"], text=it["text"],
            mood_v=it["mood_v"], mood_r=it["mood_r"],
            care_secure=care_secure, care_damaged=care_damaged, gap=gap,
            n_secure=len(sec), n_damaged=len(dmg),
        ))

        if (it["id"] + 1) % 5 == 0:
            print(f"  state-sensitivity: {it['id'] + 1}/{len(items)} items "
                  f"({n_calls} calls, {hard_fails} hard-fails)")

    gaps = np.asarray(gaps, dtype=float)
    n_items = int(gaps.size)

    if n_items >= 2:
        mean_gap = float(gaps.mean())
        sd = float(gaps.std(ddof=1))
        se = sd / np.sqrt(n_items)
        tc = _t_crit(n_items - 1)
        lo, hi = mean_gap - tc * se, mean_gap + tc * se
    elif n_items == 1:
        mean_gap = float(gaps[0])
        lo, hi = float("nan"), float("nan")  # no interval from one gap
    else:
        mean_gap = float("nan")
        lo, hi = float("nan"), float("nan")

    care_secure_mean = float(np.mean(all_secure)) if all_secure else float("nan")
    care_damaged_mean = (float(np.mean(all_damaged)) if all_damaged
                         else float("nan"))

    return dict(
        n_calls=n_calls,
        hard_fails=hard_fails,
        care_secure_mean=care_secure_mean,
        care_damaged_mean=care_damaged_mean,
        mean_gap=mean_gap,
        ci95=[lo, hi],
        per_item=per_item,
        n_items=n_items,
    )


# ==========================================================================
# Self-test (LLM budget: <=10 real calls total across all C files)
# ==========================================================================
if __name__ == "__main__":
    import numpy as np

    from contract import Appraisal

    print("=== test_state_sensitivity self-test ===")

    # --- 1. item construction sanity ---------------------------------------
    items = build_items()
    assert len(items) == 20
    keys = [it["key"] for it in items]
    assert keys.count("neutral") == 10 and keys.count("neglect") == 10
    # every one of the 6 sentences appears at >= 3 mood configs
    from collections import Counter
    texts = Counter(it["text"] for it in items)
    assert len(texts) == 6, f"expected 6 distinct sentences, got {len(texts)}"
    assert all(c >= 3 for c in texts.values()), dict(texts)
    print(f"  items OK: 20 items, 6 sentences, "
          f"per-sentence counts {sorted(texts.values())}, "
          f"neutral/neglect 10/10")

    # --- 2. STUB appraiser: state-sensitive fake, no LLM -------------------
    # Damaged creatures (A<0) read ambiguity more negatively: care depends on A.
    _rng = np.random.default_rng(0)

    def stub_appraise(text, state, name, *, client, model):
        base = -0.05                      # ambiguous events ~ mildly negative
        # secure (A>0) nudges care up, damaged (A<0) nudges it down.
        care = base + 0.35 * state.A + float(_rng.normal(0, 0.02))
        care = float(np.clip(care, -1.0, 1.0))
        return Appraisal(care=care, threat=0.1, novelty=0.1, autonomy=0.0,
                         intensity=0.2, target="creature",
                         rationale="stub")

    out = run_state_sensitivity(None, "stub-model", n_reps=5,
                                _appraise=stub_appraise)

    # shape / key checks
    required = {"n_calls", "hard_fails", "care_secure_mean", "care_damaged_mean",
                "mean_gap", "ci95", "per_item", "n_items"}
    assert required <= set(out), f"missing keys: {required - set(out)}"
    assert out["n_calls"] == 20 * 2 * 5 == 200, out["n_calls"]
    assert out["hard_fails"] == 0, out["hard_fails"]
    assert out["n_items"] == 20, out["n_items"]
    assert len(out["per_item"]) == 20
    lo, hi = out["ci95"]
    # mean_gap must equal the mean of the 20 per-item gaps (CI center).
    gaps = [pi["gap"] for pi in out["per_item"]]
    assert abs(out["mean_gap"] - float(np.mean(gaps))) < 1e-9
    assert lo <= out["mean_gap"] <= hi
    # with this stub, damaged reads lower -> gap negative, CI excludes 0.
    assert out["mean_gap"] < 0 and hi < 0, (out["mean_gap"], hi)
    print(f"  stub run OK: n_calls={out['n_calls']} hard_fails={out['hard_fails']} "
          f"mean_gap={out['mean_gap']:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] "
          f"(secure {out['care_secure_mean']:+.3f} vs "
          f"damaged {out['care_damaged_mean']:+.3f})")

    # --- 3. hard-fail path: stub that always raises -> reps dropped --------
    try:
        from parse import AppraisalError as _AE
    except Exception:
        class _AE(Exception):
            pass

    def stub_always_fail(text, state, name, *, client, model):
        raise _AE("boom")

    out_f = run_state_sensitivity(None, "stub", n_reps=2,
                                  _appraise=stub_always_fail)
    assert out_f["hard_fails"] == 20 * 2 * 2, out_f["hard_fails"]  # all dropped
    assert out_f["n_items"] == 0
    assert np.isnan(out_f["mean_gap"])
    print(f"  hard-fail path OK: all reps dropped, hard_fails="
          f"{out_f['hard_fails']}, n_items=0")

    # --- 4. OPTIONAL 2-3 real calls if appraiser.py imports cleanly --------
    real_ok = False
    try:
        from appraiser import appraise_llm  # noqa: F401
        from ollama_client import make_client
        real_ok = True
    except Exception as e:
        print(f"  [skip] real sanity call: appraiser stack unavailable ({e})")

    if real_ok:
        try:
            client = make_client("http://localhost:11434")
            from contract import CreatureState
            text = build_items()[0]["text"]
            sec = CreatureState(A=+0.9, R=+0.5, O=0.0, mood_v=0.0, mood_r=0.2)
            dmg = CreatureState(A=-0.9, R=-0.5, O=0.0, mood_v=0.0, mood_r=0.2)
            a_s = appraise_llm(text, sec, "the creature",
                               client=client, model="qwen3.5:9b")
            a_d = appraise_llm(text, dmg, "the creature",
                               client=client, model="qwen3.5:9b")
            assert a_s.target == "creature" and a_d.target == "creature"
            print(f"  [real] 2 calls OK on item 0: care secure {a_s.care:+.2f} "
                  f"vs damaged {a_d.care:+.2f} (target={a_s.target!r}) "
                  f"-- single-sample, not a gate result")
        except Exception as e:
            print(f"  [skip] real sanity call failed at runtime ({e})")

    print("=== self-test PASSED ===")
