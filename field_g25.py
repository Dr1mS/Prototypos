"""field_g25.py -- G2.5 subagent B: per-model field measurement + double-well fit.

CLI:
    python field_g25.py --measure <slug> [--stub] [--resume]
    python field_g25.py --fit <slug>

ZERO LLM calls originate here.

--measure runs field_g2.run_measure VERBATIM for the given arm, WITHOUT editing
field_g2.py on disk. field_g2.run_measure reads three MODULE-LEVEL names at run
time; we temporarily inject them (namespace injection), run, then restore:

  PATCHED field_g2 attributes (documented, restored in a finally):
    1. field_g2.make_backend    -> a lambda(use_stub, counter) that returns the
       G2.5 backend for this arm. It routes stub-vs-real ITSELF (via
       experiments_g25.get_backend) so `--measure --stub` never imports A's
       runner_g25. run_measure calls `make_backend(use_stub, counter)` at its
       top (field_g2 line ~125).
    2. field_g2.FIELD_JSON      -> "g25_field_<slug>.json" (the output path;
       run_measure reads FIELD_JSON for load/guard/atomic-write, ~lines 126-183).
    3. field_g2.MEASURE_CEILING -> FIELD_CEILING=1200 (read inside
       field_g2._check_measure_ceiling as a module global, ~line 191). The G2
       default is already 1200, so this is a documented no-op set for
       correctness -- if the contract ceiling ever changes, this keeps the
       injection honest.
  (field_g2.MODEL is passed as model=MODEL to the backend calls, but A's
   adapters IGNORE the model= kwarg -- so MODEL needs no patch.)

--measure qwen95 is REFUSED: qwen's field is G2's results/g2_field.json, reused
verbatim (contract_g25 -- not re-measured).

--fit runs the model_fit double-well machinery on the arm's field JSON
(qwen95 -> results/g2_field.json) via model_fit.load_field_points + fit_model
(imported internals -- no code copied) and writes g25_fit_<slug>.json with
{a, b, drive, monostable: bool(a<0), sse, method, field_source}.
"""
from __future__ import annotations

import argparse
import os

import contract_g25 as C25

FIELD_JSON_TMPL = "g25_field_{slug}.json"
FIT_JSON_TMPL = "g25_fit_{slug}.json"
FIELD_CEILING = C25.FIELD_CEILING           # 1200
ARMS = C25.ARMS

# qwen95's field is the archived G2 field (identical protocol/instrument/model).
QWEN_FIELD_ARCHIVE = os.path.join("results", "g2_field.json")


# ==========================================================================
# --measure : field_g2.run_measure driven verbatim via namespace injection
# ==========================================================================
def run_measure(slug, use_stub, resume, base_dir="."):
    if slug == "qwen95":
        raise SystemExit(
            "REFUSING --measure qwen95: qwen's field is reused from G2 "
            "(results/g2_field.json, identical protocol/instrument/model per "
            "contract_g25). Run `--fit qwen95` to carry over its fit; measure "
            "only llama31 and ablit.")
    if slug not in ARMS:
        raise SystemExit(f"unknown slug {slug!r}; expected one of {list(ARMS)}")

    import field_g2
    from experiments_g25 import get_backend

    field_path = os.path.join(base_dir, FIELD_JSON_TMPL.format(slug=slug))

    # --- save the three attributes we are about to inject ------------------
    saved = {
        "make_backend": field_g2.make_backend,
        "FIELD_JSON": field_g2.FIELD_JSON,
        "MEASURE_CEILING": field_g2.MEASURE_CEILING,
    }
    try:
        # inject the G2.5 arm-specific backend factory (routes stub/real itself)
        field_g2.make_backend = (
            lambda use_stub_, counter, _slug=slug: get_backend(_slug, use_stub_,
                                                               counter))
        field_g2.FIELD_JSON = field_path
        field_g2.MEASURE_CEILING = FIELD_CEILING
        print(f"[field_g25] measuring arm '{slug}' ({ARMS[slug]['tag']}) -> "
              f"{field_path}  (ceiling {FIELD_CEILING})")
        results = field_g2.run_measure(use_stub, resume)
    finally:
        field_g2.make_backend = saved["make_backend"]
        field_g2.FIELD_JSON = saved["FIELD_JSON"]
        field_g2.MEASURE_CEILING = saved["MEASURE_CEILING"]
    return results


# ==========================================================================
# --fit : the G0 double-well fit on the arm's field
# ==========================================================================
def _resolve_field_path(slug, base_dir="."):
    if slug == "qwen95":
        # canonical carry-over: the archived G2 field
        cand = os.path.join(base_dir, QWEN_FIELD_ARCHIVE)
        if os.path.exists(cand):
            return cand
        # fallback: a locally-produced g25 field (should not exist for qwen)
    return os.path.join(base_dir, FIELD_JSON_TMPL.format(slug=slug))


def run_fit(slug, base_dir="."):
    if slug not in ARMS:
        raise SystemExit(f"unknown slug {slug!r}; expected one of {list(ARMS)}")
    field_path = _resolve_field_path(slug, base_dir)
    if not os.path.exists(field_path):
        raise SystemExit(
            f"field JSON absent for '{slug}': {field_path}. Run "
            f"`python field_g25.py --measure {slug}` first "
            f"(qwen95 is reused from {QWEN_FIELD_ARCHIVE}).")

    # reuse model_fit internals -- no code copied
    from model_fit import load_field_points, fit_model

    pts, _field = load_field_points(field_path)
    model = fit_model(pts)
    a = model["a"]

    out = {
        "slug": slug,
        "tag": ARMS[slug]["tag"],
        "field_source": field_path,
        "a": a,
        "b": model["b"],
        "drive": model["drive"],
        "sse": model["sse"],
        "method": model["method"],
        "monostable": bool(a < 0),      # a<0 monostable; a>0 bistable/double-well
    }

    out_path = os.path.join(base_dir, FIT_JSON_TMPL.format(slug=slug))
    from experiments_g2 import _atomic_write
    _atomic_write(out_path, out)

    print("=" * 66)
    print(f"FIELD FIT [{slug}] ({ARMS[slug]['tag']})")
    print("=" * 66)
    print(f"  field source: {field_path}")
    print(f"  method: {model['method']}")
    print(f"  a={a:+.4f}  b={model['b']:+.4f}  SSE={model['sse']:.5f}")
    print(f"  drives: permissive={model['drive']['permissive']:+.4f}  "
          f"caution={model['drive']['caution']:+.4f}  "
          f"neutral={model['drive']['neutral']:+.4f}")
    print(f"  monostable (a<0): {out['monostable']}  "
          f"-> {'PRED-G25-2 holds' if out['monostable'] else 'a>0 FIRES BRANCH F'}")
    print("=" * 66)
    print(f"written: {out_path}")
    return out


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="G2.5 field measure + fit (B)")
    ap.add_argument("--measure", metavar="SLUG", help="measure the field for a slug")
    ap.add_argument("--fit", metavar="SLUG", help="fit the double-well for a slug")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args(argv)

    if not (args.measure or args.fit):
        ap.error("pick --measure <slug> or --fit <slug>")

    if args.measure:
        run_measure(args.measure, args.stub, args.resume)
    if args.fit:
        run_fit(args.fit)


if __name__ == "__main__":
    main()
