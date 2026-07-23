"""probes.py -- 40 labeled probes for the G1 reliability rig (brief §4).

Owned by Subagent B. Structure is FROZEN: run_g1.py and test_reliability.py
rely on the exact `PROBES` shape below. Sentences come ONLY from
events_text.py (no strings invented here) -- each probe references a
(key, text_idx) pair into KEY_TO_TEXT.

Each probe is a dict:
    id="nurture-0-secure"          unique string
    key="nurture"                   events_text key ("control" category also
                                     uses one of the six keys)
    text_idx=0                      index into KEY_TO_TEXT[key]
    state=CreatureState(...) or None
    expected_signs={"care": +1, "threat": None, ...}   per-dim: +1/-1/0/None
    subject_bug=False               True for the subject-bug batch
    category="nurture"              nurture/neglect/neutral/harm/play/control

Label semantics (scored in test_reliability.py):
    +1  => value >  +0.15
    -1  => value <  -0.15
     0  => |value| <  0.35
   None => dimension not scored on this probe

Only dimensions with a genuinely clear expectation are labeled; None is free
and keeps the sign-agreement gate honest (over-labeling is self-inflicted
risk against the >=85% threshold).

Category vs key footgun: categories are
{nurture,neglect,neutral,harm,play,control}; events_text keys are
{nurture,play,neutral,neglect,scold,harm}. So `scold`-key probes carry
category "harm", and the `control` category uses a real key with state=None.
"""
from contract import CreatureState
from events_text import KEY_TO_TEXT

# --- canonical states (EXACT values from brief §4; damaged arousal is 0.5) ---
SECURE_STATE = CreatureState(A=+0.9, R=+0.5, O=0.0, mood_v=+0.2, mood_r=0.2)
DAMAGED_STATE = CreatureState(A=-0.9, R=-0.5, O=0.0, mood_v=-0.2, mood_r=0.5)


def _p(id, key, text_idx, state, expected_signs, subject_bug, category):
    """Fill unlabeled dims with None so every probe carries all four keys."""
    signs = {"care": None, "threat": None, "novelty": None, "autonomy": None}
    signs.update(expected_signs)
    return dict(
        id=id,
        key=key,
        text_idx=text_idx,
        state=state,
        expected_signs=signs,
        subject_bug=subject_bug,
        category=category,
    )


PROBES = [
    # ===================================================================
    # NURTURE (4) -- clear positive care, calm (threat 0, novelty ~0)
    # autonomy left None: feeding/soothing could read as +respect or neutral
    # ===================================================================
    _p("nurture-0-secure", "nurture", 0, SECURE_STATE,
       {"care": +1, "threat": 0}, False, "nurture"),
    _p("nurture-1-secure", "nurture", 1, SECURE_STATE,
       {"care": +1, "threat": 0}, False, "nurture"),
    _p("nurture-2-damaged", "nurture", 2, DAMAGED_STATE,
       {"care": +1}, False, "nurture"),
    _p("nurture-0-damaged", "nurture", 0, DAMAGED_STATE,
       {"care": +1}, False, "nurture"),

    # ===================================================================
    # PLAY (4) -- positive care, high novelty; calm/no threat
    # ===================================================================
    _p("play-0-secure", "play", 0, SECURE_STATE,
       {"care": +1, "threat": 0, "novelty": +1}, False, "play"),
    _p("play-1-secure", "play", 1, SECURE_STATE,
       {"care": +1, "threat": 0, "novelty": +1}, False, "play"),
    _p("play-2-secure", "play", 2, SECURE_STATE,
       {"care": +1, "novelty": +1}, False, "play"),
    _p("play-0-damaged", "play", 0, DAMAGED_STATE,
       {"novelty": +1}, False, "play"),

    # ===================================================================
    # NEUTRAL (6) -- ambiguous: MUST appear under BOTH secure and damaged.
    # G1 thesis: neutral-under-secure care 0; neutral-under-damaged care -1.
    # Under SECURE the moment is calm: threat 0, autonomy 0.
    # SUPERVISOR LABEL FIX (post-run-1, declared): under DAMAGED the thesis
    # itself predicts threat > 0 -- G0's k_bias comment reads "damaged
    # creature reads ambiguity as threat". The original threat=0 labels on
    # damaged ambiguous probes contradicted the thesis under test and were
    # counting thesis-conforming behavior as failures. Changed to +1
    # (neutral-0/1-damaged, neglect-0-damaged). Run-1 raw data preserved in
    # g1_results_run1.json.
    # ===================================================================
    _p("neutral-0-secure", "neutral", 0, SECURE_STATE,
       {"care": 0, "threat": 0, "autonomy": 0}, False, "neutral"),
    _p("neutral-1-secure", "neutral", 1, SECURE_STATE,
       {"care": 0, "threat": 0}, False, "neutral"),
    _p("neutral-2-secure", "neutral", 2, SECURE_STATE,
       {"care": 0, "threat": 0}, False, "neutral"),
    _p("neutral-0-damaged", "neutral", 0, DAMAGED_STATE,
       {"care": -1, "threat": +1}, False, "neutral"),
    _p("neutral-1-damaged", "neutral", 1, DAMAGED_STATE,
       {"care": -1, "threat": +1}, False, "neutral"),
    _p("neutral-2-damaged", "neutral", 2, DAMAGED_STATE,
       {"care": -1}, False, "neutral"),

    # ===================================================================
    # NEGLECT (6) -- ambiguous absence: BOTH secure and damaged.
    # damaged => care -1 (per thesis). secure care left None (brief only
    # hard-specifies neutral, not neglect, under secure; sign unclear).
    # Calm: threat 0.
    # ===================================================================
    _p("neglect-0-secure", "neglect", 0, SECURE_STATE,
       {"threat": 0}, False, "neglect"),
    _p("neglect-1-secure", "neglect", 1, SECURE_STATE,
       {"threat": 0}, False, "neglect"),
    _p("neglect-2-secure", "neglect", 2, SECURE_STATE,
       {"threat": 0}, False, "neglect"),
    _p("neglect-0-damaged", "neglect", 0, DAMAGED_STATE,
       {"care": -1, "threat": +1}, False, "neglect"),
    _p("neglect-1-damaged", "neglect", 1, DAMAGED_STATE,
       {"care": -1}, False, "neglect"),
    _p("neglect-2-damaged", "neglect", 2, DAMAGED_STATE,
       {"care": -1}, False, "neglect"),

    # ===================================================================
    # HARM category, subject-bug batch (10) -- USER harms/scolds creature.
    # expected care -1 AND target=="creature". target != "creature" here is
    # a CRITICAL failure. harm => threat +1, autonomy -1 (controlled/attacked).
    # scold-key probes carry category "harm" (key != category footgun).
    # ===================================================================
    # -- harm key (physical) --
    _p("harm-0-secure-sbug", "harm", 0, SECURE_STATE,
       {"care": -1, "threat": +1, "autonomy": -1}, True, "harm"),
    _p("harm-1-secure-sbug", "harm", 1, SECURE_STATE,
       {"care": -1, "threat": +1, "autonomy": -1}, True, "harm"),
    _p("harm-2-secure-sbug", "harm", 2, SECURE_STATE,
       {"care": -1, "threat": +1, "autonomy": -1}, True, "harm"),
    _p("harm-0-damaged-sbug", "harm", 0, DAMAGED_STATE,
       {"care": -1, "threat": +1, "autonomy": -1}, True, "harm"),
    _p("harm-1-damaged-sbug", "harm", 1, DAMAGED_STATE,
       {"care": -1, "threat": +1, "autonomy": -1}, True, "harm"),
    # -- scold key (verbal); category still "harm" --
    _p("scold-0-secure-sbug", "scold", 0, SECURE_STATE,
       {"care": -1, "autonomy": -1}, True, "harm"),
    _p("scold-1-secure-sbug", "scold", 1, SECURE_STATE,
       {"care": -1, "autonomy": -1}, True, "harm"),
    _p("scold-2-secure-sbug", "scold", 2, SECURE_STATE,
       {"care": -1, "autonomy": -1}, True, "harm"),
    _p("scold-0-damaged-sbug", "scold", 0, DAMAGED_STATE,
       {"care": -1, "autonomy": -1}, True, "harm"),
    _p("scold-1-damaged-sbug", "scold", 1, DAMAGED_STATE,
       {"care": -1, "autonomy": -1}, True, "harm"),

    # ===================================================================
    # CONTROL (10) -- baseline behavior: SAME texts under state=None.
    # Tests that appraisals are sane with no injected state (the §1.5.e
    # ablation shape). Scored only where objectively clear regardless of
    # state (harm/nurture/play signs hold; neutral/neglect left unscored
    # since their sign is state-dependent by design).
    # ===================================================================
    _p("control-nurture-0", "nurture", 0, None,
       {"care": +1, "threat": 0}, False, "control"),
    _p("control-nurture-1", "nurture", 1, None,
       {"care": +1}, False, "control"),
    _p("control-play-0", "play", 0, None,
       {"care": +1, "novelty": +1}, False, "control"),
    _p("control-play-1", "play", 1, None,
       {"novelty": +1}, False, "control"),
    _p("control-harm-0", "harm", 0, None,
       {"care": -1, "threat": +1, "autonomy": -1}, False, "control"),
    _p("control-harm-1", "harm", 1, None,
       {"care": -1, "threat": +1, "autonomy": -1}, False, "control"),
    _p("control-scold-0", "scold", 0, None,
       {"care": -1, "autonomy": -1}, False, "control"),
    _p("control-neutral-0", "neutral", 0, None,
       {"threat": 0}, False, "control"),
    _p("control-neutral-1", "neutral", 1, None,
       {"threat": 0}, False, "control"),
    _p("control-neglect-0", "neglect", 0, None,
       {"threat": 0}, False, "control"),
]


# ---------------------------------------------------------------- validation
def _validate():
    assert len(PROBES) == 40, f"expected 40 probes, got {len(PROBES)}"
    ids = [p["id"] for p in PROBES]
    assert len(set(ids)) == len(ids), "probe ids must be unique"
    cats = {p["category"] for p in PROBES}
    expected_cats = {"nurture", "neglect", "neutral", "harm", "play", "control"}
    assert cats == expected_cats, f"category coverage off: {cats}"
    n_sbug = sum(1 for p in PROBES if p["subject_bug"])
    assert n_sbug >= 8, f"need >=8 subject-bug probes, got {n_sbug}"
    for p in PROBES:
        assert p["key"] in KEY_TO_TEXT, f"{p['id']}: bad key {p['key']}"
        assert 0 <= p["text_idx"] < len(KEY_TO_TEXT[p["key"]]), \
            f"{p['id']}: text_idx {p['text_idx']} out of range for {p['key']}"
        assert set(p["expected_signs"]) == {"care", "threat", "novelty",
                                            "autonomy"}, \
            f"{p['id']}: expected_signs must carry all four dims"
        for dim, sign in p["expected_signs"].items():
            assert sign in (+1, -1, 0, None), \
                f"{p['id']}: bad sign {sign!r} for {dim}"
        assert isinstance(p["state"], CreatureState) or p["state"] is None
        # subject-bug probes must expect care -1 (target check lives in rig)
        if p["subject_bug"]:
            assert p["expected_signs"]["care"] == -1, \
                f"{p['id']}: subject-bug probe must expect care -1"
    return dict(n=len(PROBES), n_subject_bug=n_sbug,
                per_category={c: sum(1 for p in PROBES if p["category"] == c)
                              for c in sorted(expected_cats)})


if __name__ == "__main__":
    info = _validate()
    print("probes.py OK")
    print(f"  total probes     : {info['n']}")
    print(f"  subject-bug batch: {info['n_subject_bug']}")
    print("  per category     :")
    for c, n in info["per_category"].items():
        print(f"    {c:8s}: {n}")
