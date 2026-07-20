"""seam.py -- supervisor-owned adapter between the G0 harness and the real
LLM appraiser (G1 brief §1.5.c, copied verbatim).

Appraiser interface: appraiser(ev, s, age, p, bias_on) -> (pv, pa, nov)

Double-bias guard (§1.5.d): on the real path the LLM does the
perception-biasing (it receives the creature state). The adapter uses a.care
raw and NEVER re-applies k_bias. The real path touches p["k_bias"] nowhere.

Real-path ablation (§1.5.e): bias_on toggles the *mock's* numeric bias and is
meaningless on the real path. The real-model equivalent of "bias OFF" is
inject_state=False -- the appraiser prompt omits the creature's state.
"""
import numpy as np

from contract import CreatureState, Appraisal
from G0 import mock_appraiser  # noqa: F401  (re-export; mock path lives in G0)

_TEXT_RNG = np.random.default_rng(11)


def _pick_random(texts):
    return texts[int(_TEXT_RNG.integers(len(texts)))]


def appraisal_to_pvpanov(a: Appraisal):
    # EXACTLY the mock's second half with pcare = care (LLM already biased).
    pv = float(np.clip(a.care - 0.5*a.threat + 0.3*a.autonomy, -1.0, 1.0))
    pa = float(np.clip(0.7*a.threat + 0.5*a.novelty, 0.0, 1.0))
    return pv, pa, a.novelty


def make_real_appraiser(client, model, key_to_text, name="the creature",
                        text_pick="first", inject_state=True):
    from appraiser import appraise_llm  # lazy: subagent A's module

    def real_appraiser(ev, s, age, p, bias_on):
        state = CreatureState(A=s[2], R=s[3], O=s[4],
                              mood_v=s[0], mood_r=s[1])
        texts = key_to_text[ev]
        text = texts[0] if text_pick == "first" else _pick_random(texts)
        # inject_state=False is the real-path "bias OFF" ablation: omit the
        # creature's state from the prompt entirely (see §1.5.e).
        a = appraise_llm(text, state if inject_state else None,
                         name, client=client, model=model)
        # k_bias is DELIBERATELY NOT applied here. Double-bias guard (§1.5.d).
        return appraisal_to_pvpanov(a)

    return real_appraiser
