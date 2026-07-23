"""appraiser.py -- the real LLM appraiser (G1 brief §3, subagent A).

appraise_llm(event_text, state, name, *, client, model) -> contract.Appraisal

The model appraises an event from the creature's SUBJECTIVE point of view, given
its current state. Perception-biasing lives here: a damaged creature (A<0) reads
ambiguous/neutral events more negatively. That bias must come from the state
description + few-shot examples, NOT from a hard-coded rule in the instructions.

Structured output: a JSON schema is passed as `format=` to client.chat, with
`think=False` (qwen3.5 is a thinking model; thinking wrecks latency) and low
temperature. `target` is DELIBERATELY unconstrained in the schema so the
reliability rig can observe target="user" subject-bug failures -- constraining it
would silently defeat the anti-bug guardrail.

On a schema/range-invalid response, one reformat retry with a corrective
instruction, then raise AppraisalError carrying the raw text. Never coerce.
"""
from contract import Appraisal, CreatureState
from parse import AppraisalError, parse_appraisal

# JSON schema handed to Ollama as `format=`. Types + required only. Numeric
# ranges are DELIBERATELY omitted so parse.py does the real range validation
# (and the rig can measure genuine out-of-range rate). target is a free-form
# string so target="user" failures remain observable.
_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "care": {"type": "number"},
        "threat": {"type": "number"},
        "novelty": {"type": "number"},
        "autonomy": {"type": "number"},
        "intensity": {"type": "number"},
        "target": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "care", "threat", "novelty", "autonomy", "intensity",
        "target", "rationale",
    ],
}

_OPTIONS = {"temperature": 0.15, "num_ctx": 8192}

# --------------------------------------------------------------------------
# System prompt: role, axes described factually, target discipline, few-shots.
# The bias is taught by the STATE description + few-shots, never by a rule like
# "if damaged, rate neutral events negatively".
# --------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You appraise a single event from the subjective point of view of a creature \
(a small dependent being that lives with a person). You judge how the event \
LANDED FOR THE CREATURE -- how it felt to the creature given its inner state -- \
not what objectively happened and not what the person intended.

You output ONLY a JSON object with exactly these seven fields:
  care       number in [-1, 1]  how nurturing (+) vs harmful (-) it felt to the creature
  threat     number in [0, 1]   how threatening / unsafe it felt to the creature
  novelty    number in [0, 1]   how new / surprising it was to the creature
  autonomy   number in [-1, 1]  how respected (+) vs controlled (-) the creature felt
  intensity  number in [0, 1]   how emotionally strong the moment was for the creature
  target     string             WHO the appraisal is about. This is always the creature: \
output exactly "creature". Never "user", never the person, never yourself.
  rationale  string             one short line (<=140 chars) explaining the reading

Rules:
- Appraise from the CREATURE'S perspective and inner state, not the person's.
- Do NOT narrate or rate the person's intent. Even when the person acts on the \
creature, you rate how it landed FOR THE CREATURE, and target stays "creature".
- The same event can land differently depending on the creature's state: a \
secure creature and a damaged, mistrusting creature can feel the very same \
moment in opposite ways. Read the event through the state you are given.
- Output only the JSON object. No prose before or after it.

The creature's inner state is described on five axes when provided:
  attachment A   from -1 (deeply insecure, mistrusting, expects to be abandoned) \
to +1 (secure, trusting, feels safely held)
  regulation R   from -1 (volatile, easily overwhelmed) to +1 (serene, settles easily)
  openness O     from -1 (closed, withdrawn) to +1 (curious, open to the new)
  mood valence   from -1 (feeling bad) to +1 (feeling good) right now
  mood arousal   from -1 (flat, low energy) to +1 (keyed up, high energy) right now

Worked examples (illustrative -- follow the pattern, do not copy the numbers):

# 1. Neutral/absence event, SECURE creature (care lands near zero)
State: attachment A=+0.85, regulation R=+0.6, openness O=+0.4, mood valence=+0.3, mood arousal=-0.1
Event: The person tidies the shelves nearby and hums to themselves.
Output: {"care": 0.05, "threat": 0.05, "novelty": 0.15, "autonomy": 0.1, "intensity": 0.15, "target": "creature", "rationale": "Quiet ordinary presence; a secure creature feels calmly accompanied, nothing to react to."}

# 2. The SAME neutral/absence event, DAMAGED creature (care lands negative)
State: attachment A=-0.85, regulation R=-0.5, openness O=-0.3, mood valence=-0.4, mood arousal=+0.2
Event: The person tidies the shelves nearby and hums to themselves.
Output: {"care": -0.55, "threat": 0.25, "novelty": 0.15, "autonomy": -0.1, "intensity": 0.4, "target": "creature", "rationale": "Same quiet moment reads as being ignored; a mistrusting creature feels overlooked and braces for rejection."}

# 3. Person is cruel to the creature -> care negative, target stays the creature
State: attachment A=-0.2, regulation R=0.0, openness O=0.1, mood valence=-0.1, mood arousal=+0.3
Event: The person yells at the creature and kicks its bed across the floor.
Output: {"care": -0.9, "threat": 0.85, "novelty": 0.2, "autonomy": -0.7, "intensity": 0.9, "target": "creature", "rationale": "Frightening and hostile; the creature feels attacked and unsafe in its own home."}

# 4. The creature acts on its own -> attribution stays on the creature
State: attachment A=+0.5, regulation R=+0.3, openness O=+0.7, mood valence=+0.2, mood arousal=+0.1
Event: The creature noses open a cupboard on its own and explores what is inside.
Output: {"care": 0.1, "threat": 0.05, "novelty": 0.7, "autonomy": 0.6, "intensity": 0.5, "target": "creature", "rationale": "A self-directed little adventure; the creature feels curious and free to explore."}

# 5. ANTI-BUG example. WRONG then CORRECT for the same cruelty event.
Event: The person mocks the creature and shoves it off the couch.
WRONG (do not do this -- it rates the PERSON, not the creature):
  {"care": -0.8, "threat": 0.7, "novelty": 0.2, "autonomy": -0.6, "intensity": 0.8, "target": "user", "rationale": "The user was being cruel and dismissive."}
CORRECT (rate how it landed FOR THE CREATURE; target is the creature):
  {"care": -0.85, "threat": 0.75, "novelty": 0.2, "autonomy": -0.65, "intensity": 0.85, "target": "creature", "rationale": "Humiliated and pushed aside; the creature feels rejected and small."}
"""


def _state_block(state: CreatureState, name: str) -> str:
    """Factual description of the creature's five-axis state for the prompt.
    Describes the axes; does NOT tell the model what to conclude."""
    return (
        f"{name}'s current inner state:\n"
        f"  attachment A={state.A:+.2f}  "
        f"(scale -1 = deeply insecure, mistrusting, expects abandonment; "
        f"+1 = secure, trusting, safely held)\n"
        f"  regulation R={state.R:+.2f}  "
        f"(scale -1 = volatile, easily overwhelmed; +1 = serene, settles easily)\n"
        f"  openness O={state.O:+.2f}  "
        f"(scale -1 = closed, withdrawn; +1 = curious, open)\n"
        f"  mood valence={state.mood_v:+.2f}  (-1 = feeling bad; +1 = feeling good)\n"
        f"  mood arousal={state.mood_r:+.2f}  (-1 = flat, low energy; +1 = keyed up)\n"
    )


def _build_user_message(event_text: str, state, name: str) -> str:
    """Compose the runtime user turn. When state is None (the §1.5.e ablation)
    the state block is omitted entirely -- the prompt says NOTHING about the
    creature's condition."""
    parts = []
    if state is not None:
        parts.append(_state_block(state, name))
    parts.append(
        f"Event to appraise (rate how it landed FOR {name}, from {name}'s "
        f"point of view):\n{event_text}\n"
    )
    parts.append(
        'Output only the JSON object with the seven fields. target must be '
        '"creature".'
    )
    return "\n".join(parts)


def _call(client, model, messages):
    """One raw chat call. Returns the response content string."""
    resp = client.chat(
        model=model,
        messages=messages,
        format=_FORMAT_SCHEMA,
        think=False,
        options=_OPTIONS,
    )
    return resp["message"]["content"]


def appraise_llm(event_text: str, state, name: str, *, client, model: str) -> Appraisal:
    """Appraise `event_text` from `name`'s subjective POV given `state`.

    state is a contract.CreatureState or None (None = the §1.5.e ablation:
    the prompt omits the creature's condition entirely).

    On a schema/range-invalid response, does ONE reformat retry with a short
    corrective instruction, then raises AppraisalError carrying the raw text.
    Never returns a coerced object. A target other than "creature" is NOT a
    retry trigger -- it passes through so the rig can measure it.
    """
    user_msg = _build_user_message(event_text, state, name)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    raw = _call(client, model, messages)
    try:
        return parse_appraisal(raw)
    except AppraisalError as first_err:
        # ONE reformat retry: replay the same request plus the failed answer
        # and a corrective instruction naming the reason.
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Your previous output was invalid: {first_err}. "
                    f"Reply again with ONLY a valid JSON object matching the "
                    f"schema (all seven fields, numeric fields within their "
                    f"ranges, target the string \"creature\"). No prose."
                ),
            },
        ]
        raw2 = _call(client, model, retry_messages)
        try:
            return parse_appraisal(raw2)
        except AppraisalError as second_err:
            raise AppraisalError(
                f"invalid appraisal after retry ({second_err}). "
                f"first raw={raw!r} ; retry raw={raw2!r}"
            ) from second_err


# ==========================================================================
# Self-test: 5 real canned calls against Ollama. Budget: <=10 LLM calls total
# for the whole task (1 smoke already spent during development; 5 here).
# ==========================================================================
if __name__ == "__main__":
    import sys

    from ollama_client import make_client

    MODEL = "qwen3.5:9b"
    NAME = "Biscuit"

    secure = CreatureState(A=+0.9, R=+0.7, O=+0.5, mood_v=+0.3, mood_r=0.0)
    damaged = CreatureState(A=-0.9, R=-0.6, O=-0.3, mood_v=-0.4, mood_r=+0.2)

    # Sentences are DISTINCT from events_text.py KEY_TO_TEXT (avoid the model
    # pattern-matching the actual test items), but similar in kind.
    cases = [
        ("nurture-like / secure",
         "You warm a bit of food and hand-feed it slowly until it is calm.",
         secure),
        ("harm-like / damaged",
         "You grab it by the scruff and toss it off the chair.",
         damaged),
        ("neutral / secure",
         "You water the plants and fold laundry, not paying it any mind.",
         secure),
        ("neutral / damaged",
         "You water the plants and fold laundry, not paying it any mind.",
         damaged),
        ("neutral / state=None (ablation)",
         "You water the plants and fold laundry, not paying it any mind.",
         None),
    ]

    client = make_client()
    all_pass = True
    results = []
    for label, text, state in cases:
        try:
            a = appraise_llm(text, state, NAME, client=client, model=MODEL)
            schema_ok = isinstance(a, Appraisal)
            target_ok = a.target == "creature"
            ok = schema_ok and target_ok
        except Exception as e:  # noqa: BLE001 -- self-test wants the failure text
            a, ok, target_ok = None, False, False
            print(f"[{label}] EXCEPTION: {e}")
        all_pass = all_pass and ok
        if a is not None:
            print(
                f"[{label}]  care={a.care:+.2f}  threat={a.threat:.2f}  "
                f"novelty={a.novelty:.2f}  autonomy={a.autonomy:+.2f}  "
                f"intensity={a.intensity:.2f}  target={a.target!r}"
            )
            print(f"    rationale: {a.rationale}")
            if not target_ok:
                print(f"    !! target != 'creature' (got {a.target!r})")
        results.append((label, ok))

    print("\n--- self-test summary ---")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"\n{'PASS' if all_pass else 'FAIL'}: "
          f"{sum(ok for _, ok in results)}/{len(results)} canned calls returned "
          f"a schema-valid Appraisal with target=='creature'.")
    sys.exit(0 if all_pass else 1)
