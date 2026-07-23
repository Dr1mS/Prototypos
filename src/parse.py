"""parse.py -- parse + validate raw LLM output into a contract.Appraisal.

G1 brief §3 / subagent A. Validation rules (verbatim from the task brief):

- Parse JSON -> all 7 fields present, numeric fields are numbers within their
  contract ranges (care [-1,1], threat [0,1], novelty [0,1], autonomy [-1,1],
  intensity [0,1]), target is a str, rationale is a str. Out-of-range numeric
  or missing field => INVALID. Never silently coerce/clamp numeric values.
- ONLY permitted coercion: a rationale longer than 140 chars may be truncated
  to 140 (it is a log string, not data).
- target != "creature" is NOT a parse error and NOT a retry trigger -- it must
  pass through so the reliability rig can measure the subject-bug rate. Only
  schema/range violations are INVALID.

On any violation, `parse_appraisal` raises AppraisalError(<reason>); the caller
(appraiser.appraise_llm) uses <reason> in its single reformat retry.
"""
import json
import math

from contract import Appraisal


class AppraisalError(Exception):
    """Raised when raw LLM output cannot be validated against the Appraisal
    contract (missing field, wrong type, or out-of-range numeric)."""


_RATIONALE_MAX = 140

# field -> inclusive (lo, hi) range
_NUMERIC_RANGES = {
    "care": (-1.0, 1.0),
    "threat": (0.0, 1.0),
    "novelty": (0.0, 1.0),
    "autonomy": (-1.0, 1.0),
    "intensity": (0.0, 1.0),
}
_ALL_FIELDS = list(_NUMERIC_RANGES) + ["target", "rationale"]


def _check_number(name, value, lo, hi):
    """Return a validated float for a numeric field, or raise AppraisalError.

    Rejects bools (bool is a subclass of int, so `True` would sail through as
    1.0), stringified numbers ("0.5"), non-finite values (NaN/Inf), and
    out-of-range values. No clamping -- out of range is INVALID.
    """
    if isinstance(value, bool):
        raise AppraisalError(f"field '{name}' is a bool, expected a number")
    if not isinstance(value, (int, float)):
        raise AppraisalError(
            f"field '{name}' is {type(value).__name__}, expected a number"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise AppraisalError(f"field '{name}' is non-finite ({value!r})")
    if not (lo <= fv <= hi):
        raise AppraisalError(
            f"field '{name}'={fv} out of range [{lo}, {hi}]"
        )
    return fv


def parse_appraisal(raw: str) -> Appraisal:
    """Parse a raw JSON string into a validated Appraisal.

    Raises AppraisalError with a short human-readable reason on any schema or
    range violation. The reason string is fed into the retry instruction by the
    caller, so keep it terse and specific.
    """
    if not isinstance(raw, str):
        raise AppraisalError(f"raw output is {type(raw).__name__}, expected str")

    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise AppraisalError(f"not valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise AppraisalError(
            f"top-level JSON is {type(obj).__name__}, expected an object"
        )

    missing = [f for f in _ALL_FIELDS if f not in obj]
    if missing:
        raise AppraisalError(f"missing field(s): {', '.join(missing)}")

    # numeric fields: type + finiteness + range, no coercion
    nums = {
        name: _check_number(name, obj[name], lo, hi)
        for name, (lo, hi) in _NUMERIC_RANGES.items()
    }

    # target: must be a str; value is NOT validated here (target != "creature"
    # passes through so the rig can measure it -- brief §3).
    target = obj["target"]
    if not isinstance(target, str):
        raise AppraisalError(
            f"field 'target' is {type(target).__name__}, expected str"
        )

    rationale = obj["rationale"]
    if not isinstance(rationale, str):
        raise AppraisalError(
            f"field 'rationale' is {type(rationale).__name__}, expected str"
        )
    # ONLY permitted coercion: truncate an over-long rationale (it is a log
    # string). Numeric fields are never coerced.
    if len(rationale) > _RATIONALE_MAX:
        rationale = rationale[:_RATIONALE_MAX]

    return Appraisal(
        care=nums["care"],
        threat=nums["threat"],
        novelty=nums["novelty"],
        autonomy=nums["autonomy"],
        intensity=nums["intensity"],
        target=target,
        rationale=rationale,
    )
