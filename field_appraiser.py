"""field_appraiser.py -- Subagent E (G1.5 §3).

A field-driven appraiser that plugs into the G0 seam with ZERO LLM calls. It
replays the appraisal *field* that subagent D measured (`field_A.json`): for
each event key and each appraisal dimension, D recorded mean+stdev over an
injected-attachment grid A in {-1, -0.5, 0, +0.5, +1}. This module linearly
interpolates that field over A, optionally injects the measured per-cell
gaussian noise, and routes the result through the EXISTING seam adapter -- it
never re-derives the pv/pa/nov math (anti-duplication, brief §7).

Documented `field_A.json` schema (coded against this, not against D's impl):

    {
      "A_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
      "field": {
        "<key>": {                      # 6 keys: nurture play neutral
                                        #          neglect scold harm
          "care":     {"mean": [5], "std": [5]},
          "threat":   {"mean": [5], "std": [5]},
          "novelty":  {"mean": [5], "std": [5]},
          "autonomy": {"mean": [5], "std": [5]}
        }, ...
      }, ...
    }

`A_values` may be absent; we then fall back to the frozen grid. Extra top-level
keys (asymmetry metrics etc.) are ignored. `intensity` is fixed at 0.5 (prereg);
if D happens to also record an intensity dim we ignore it -- the seam adapter
does not use intensity, so this is inert.

Contract ranges enforced after noise (contract.py): care, autonomy in [-1, 1];
threat, novelty (and intensity) in [0, 1].
"""

import json

import numpy as np

from contract import Appraisal
from seam import appraisal_to_pvpanov

# Frozen grid (prereg §"Frozen design parameters"). Used only if field_A.json
# omits "A_values".
DEFAULT_A_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]

# The four appraisal dims the field carries. intensity is fixed, target/rationale
# are constants -> not part of the measured field.
_DIMS = ("care", "threat", "novelty", "autonomy")

# Per-dim contract clip ranges (contract.py docstring / dataclass comments).
_CLIP = {
    "care": (-1.0, 1.0),
    "autonomy": (-1.0, 1.0),
    "threat": (0.0, 1.0),
    "novelty": (0.0, 1.0),
}

# The 6 harness event keys (events_text / G0.EVENTS). A complete field must
# cover all six.
_REQUIRED_KEYS = ("nurture", "play", "neutral", "neglect", "scold", "harm")


def _load_field(field_path_or_dict):
    """Accept either a path to field_A.json or an already-parsed dict.
    Returns (field_dict, A_values) with A_values validated ascending."""
    if isinstance(field_path_or_dict, dict):
        raw = field_path_or_dict
    else:
        with open(field_path_or_dict, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

    # The documented top-level wrapper has "field" + optional "A_values". Some
    # callers (and our own stub) may pass the bare 6-key field dict directly;
    # support both.
    if "field" in raw and isinstance(raw["field"], dict):
        field = raw["field"]
        a_values = raw.get("A_values", DEFAULT_A_VALUES)
    else:
        field = raw
        a_values = DEFAULT_A_VALUES

    a_values = [float(a) for a in a_values]
    if list(a_values) != sorted(a_values):
        raise ValueError(f"A_values must be ascending for np.interp: {a_values}")
    return field, a_values


def validate_field(field_path_or_dict):
    """Defensive completeness check against the documented schema. Raises
    ValueError with a precise message on the first problem; returns
    (field, A_values) on success. Used by experiments_field's full-scale gate."""
    field, a_values = _load_field(field_path_or_dict)
    n = len(a_values)
    for key in _REQUIRED_KEYS:
        if key not in field:
            raise ValueError(f"field missing event key {key!r}")
        cell = field[key]
        for dim in _DIMS:
            if dim not in cell:
                raise ValueError(f"field[{key!r}] missing dim {dim!r}")
            for stat in ("mean", "std"):
                if stat not in cell[dim]:
                    raise ValueError(
                        f"field[{key!r}][{dim!r}] missing {stat!r}")
                vals = cell[dim][stat]
                if len(vals) != n:
                    raise ValueError(
                        f"field[{key!r}][{dim!r}][{stat!r}] has "
                        f"{len(vals)} entries, expected {n}")
    return field, a_values


def make_field_appraiser(field_path_or_dict, noise=True, seed=31):
    """Build an appraiser(ev, s, age, p, bias_on) -> (pv, pa, nov) that replays
    the measured appraisal field.

    Parameters
    ----------
    field_path_or_dict : str | dict
        Path to `field_A.json`, or an already-parsed field/dict (schema above).
    noise : bool
        If True (default) add gaussian N(0, interp(std)) per dim, using this
        appraiser's OWN rng (default_rng(seed)); if False, use interpolated
        means only.
    seed : int
        Seed for the private noise rng (prereg default 31).

    The appraiser IGNORES `bias_on` (meaningless on the field path -- D already
    baked the state-dependence into the A grid) and never touches p["k_bias"].
    """
    field, a_values = validate_field(field_path_or_dict)
    xp = np.asarray(a_values, dtype=float)

    # Pre-extract mean/std arrays per key/dim so the hot loop is just np.interp.
    means = {k: {d: np.asarray(field[k][d]["mean"], float) for d in _DIMS}
             for k in _REQUIRED_KEYS}
    stds = {k: {d: np.asarray(field[k][d]["std"], float) for d in _DIMS}
            for k in _REQUIRED_KEYS}

    rng = np.random.default_rng(seed)

    def field_appraiser(ev, s, age, p, bias_on):
        # A is the attachment trait (state index 2 in G0's [v, r, A, R, O]).
        A = float(s[2])
        vals = {}
        for dim in _DIMS:
            m = float(np.interp(A, xp, means[ev][dim]))
            if noise:
                sd = float(np.interp(A, xp, stds[ev][dim]))
                if sd > 0.0:
                    m += float(rng.normal(0.0, sd))
            lo, hi = _CLIP[dim]
            vals[dim] = float(np.clip(m, lo, hi))

        # ANTI-DUPLICATION: build the contract Appraisal and let the seam do the
        # pv/pa/nov adapter math. intensity fixed at 0.5 (prereg).
        a = Appraisal(care=vals["care"], threat=vals["threat"],
                      novelty=vals["novelty"], autonomy=vals["autonomy"],
                      intensity=0.5, target="creature", rationale="field")
        return appraisal_to_pvpanov(a)

    return field_appraiser


# --------------------------------------------------------------------------
# Self-test (synthetic field, no LLM, no field_A.json needed)
# --------------------------------------------------------------------------
def _synthetic_field(seed=0):
    """A small but schema-complete synthetic field for tests. care declines
    with damage on ambiguous events so the appraiser behaves plausibly; exact
    numbers are irrelevant -- we only test wiring, ranges, determinism."""
    rng = np.random.default_rng(seed)
    a_vals = DEFAULT_A_VALUES
    # rough per-key base appraisal primitives (mirrors G0.EVENTS intent)
    base = {
        "nurture": dict(care=+1.0, threat=0.0, novelty=0.1, autonomy=0.0),
        "play":    dict(care=+0.6, threat=0.0, novelty=0.7, autonomy=+0.3),
        "neutral": dict(care=+0.0, threat=0.0, novelty=0.1, autonomy=0.0),
        "neglect": dict(care=-0.5, threat=0.0, novelty=0.0, autonomy=0.0),
        "scold":   dict(care=-0.7, threat=0.4, novelty=0.1, autonomy=-0.6),
        "harm":    dict(care=-1.0, threat=0.9, novelty=0.3, autonomy=-0.8),
    }
    field = {}
    for k, prim in base.items():
        cell = {}
        for d in _DIMS:
            lo, hi = _CLIP[d]
            means = []
            for A in a_vals:
                # ambiguity read-down under damage for care (illustrative only)
                v = prim[d]
                if d == "care":
                    amb = 1.0 - abs(prim["care"])
                    v = prim["care"] - 0.6 * max(0.0, -A) * amb
                means.append(float(np.clip(v, lo, hi)))
            cell[d] = {"mean": means,
                       "std": [float(abs(rng.normal(0, 0.03))) for _ in a_vals]}
        field[k] = cell
    return {"A_values": a_vals, "field": field}


if __name__ == "__main__":
    print("=== field_appraiser self-test (no LLM) ===")
    fld = _synthetic_field()

    # validate schema
    validate_field(fld)
    print("  schema validation OK (6 keys x 4 dims x 5-cell mean+std)")

    # noise ON is deterministic given the seed
    a1 = make_field_appraiser(fld, noise=True, seed=31)
    a2 = make_field_appraiser(fld, noise=True, seed=31)
    s = np.array([0.0, 0.0, -0.5, 0.0, 0.0])  # damaged-ish state
    out1 = [a1(k, s, 0, {}, True) for k in _REQUIRED_KEYS]
    out2 = [a2(k, s, 0, {}, True) for k in _REQUIRED_KEYS]
    assert out1 == out2, "same seed must be reproducible"
    print("  determinism (seed=31) OK")

    # every output respects the pv/pa/nov ranges (pv in [-1,1], pa in [0,1])
    for k in _REQUIRED_KEYS:
        pv, pa, nov = a1(k, s, 0, {}, True)
        assert -1.0 <= pv <= 1.0, (k, pv)
        assert 0.0 <= pa <= 1.0, (k, pa)
        assert 0.0 <= nov <= 1.0, (k, nov)
    print("  output ranges OK (pv in [-1,1], pa,nov in [0,1])")

    # noise OFF == interpolated means; interp at a grid point == that cell mean
    a_off = make_field_appraiser(fld, noise=False)
    s0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # A=0 -> exact grid cell index 2
    exp = Appraisal(care=fld["field"]["neutral"]["care"]["mean"][2],
                    threat=fld["field"]["neutral"]["threat"]["mean"][2],
                    novelty=fld["field"]["neutral"]["novelty"]["mean"][2],
                    autonomy=fld["field"]["neutral"]["autonomy"]["mean"][2],
                    intensity=0.5, target="creature", rationale="field")
    assert a_off("neutral", s0, 0, {}, True) == appraisal_to_pvpanov(exp)
    print("  noise-OFF == seam(interp means) at grid point OK")

    # bias_on is ignored (field path); toggling it changes nothing
    a3 = make_field_appraiser(fld, noise=False)
    assert (a3("harm", s, 0, {}, True) == a3("harm", s, 0, {}, False))
    print("  bias_on ignored OK")

    print("=== field_appraiser self-test PASSED ===")
