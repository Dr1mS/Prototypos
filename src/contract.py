"""contract.py -- single source of truth, imported everywhere, redefined nowhere.

FROZEN (G1 brief §1). No subagent may alter this schema. Change needed ->
escalate to the supervisor, never edit unilaterally.

Semantic contract: care/threat/novelty/autonomy are the creature's *subjective*
appraisal given its state -- not an objective description. The model is told the
creature's state and asked how the event *landed for it*. That is where
perception-biasing lives.

`target` catches the classic extractor bug (attributing to the user instead of
the creature). target != "creature" is a hard fail the rig logs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CreatureState:
    A: float      # attachment  secure(+1) .. damaged(-1)
    R: float      # regulation  serene(+1) .. volatile(-1)
    O: float      # openness
    mood_v: float # current mood valence
    mood_r: float # current mood arousal


@dataclass(frozen=True)
class Appraisal:
    care: float       # [-1,1]  SUBJECTIVE: how nurturing/harmful it felt TO the creature
    threat: float     # [0,1]
    novelty: float    # [0,1]
    autonomy: float   # [-1,1]  respected(+) .. controlled(-)
    intensity: float  # [0,1]
    target: str       # MUST be "creature". anti-bug guardrail.
    rationale: str    # <=140 chars, one line (debugging + the shareable log)


# signature the appraiser MUST satisfy:
# def appraise_llm(event_text: str, state: CreatureState, name: str,
#                  *, client, model: str) -> Appraisal
